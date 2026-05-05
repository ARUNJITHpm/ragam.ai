from __future__ import annotations

import re
from pathlib import Path

from pydub import AudioSegment, silence

from malayalam_tts_dataset.asr.whisper_transcriber import TranscriptSegment
from malayalam_tts_dataset.audio.validation import validate_audio_file, validate_audio_segment
from malayalam_tts_dataset.dataset.records import DatasetRecord


SENTENCE_END_RE = re.compile(r"[.!?।॥。]+$")


def segment_transcribed_audio(
    *,
    trimmed_wav_path: str | Path,
    transcript_segments: list[TranscriptSegment],
    video_id: str,
    source_url: str,
    output_segments_dir: str | Path,
    min_utterance_seconds: float,
    max_utterance_seconds: float,
    silence_thresh_dbfs: float,
    min_silence_len_ms: int,
    keep_silence_ms: int,
    overwrite: bool = False,
) -> list[DatasetRecord]:
    """
    Convert trimmed audio + Whisper transcript segments into TTS utterance clips.

    Strategy:
    1. Use Whisper segments as primary semantic/time boundaries.
    2. Merge short segments until minimum duration is reached.
    3. Avoid exceeding maximum duration.
    4. Prefer flushing at sentence punctuation when duration is valid.
    5. Use silence detection as secondary helper to slightly improve cut boundaries.
    6. Validate exported utterances.

    Args:
        trimmed_wav_path:
            Trimmed WAV path.
        transcript_segments:
            Whisper transcript segments.
        video_id:
            YouTube video ID.
        source_url:
            Source YouTube URL.
        output_segments_dir:
            Directory to save utterance WAV files.
        min_utterance_seconds:
            Minimum utterance duration.
        max_utterance_seconds:
            Maximum utterance duration.
        silence_thresh_dbfs:
            Silence threshold for pydub.
        min_silence_len_ms:
            Minimum silence length.
        keep_silence_ms:
            Silence padding around exported clips.
        overwrite:
            Whether to overwrite existing utterance WAV files.

    Returns:
        List of DatasetRecord.
    """
    input_path = Path(trimmed_wav_path).expanduser().resolve()
    output_dir = Path(output_segments_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if min_utterance_seconds <= 0:
        raise ValueError("min_utterance_seconds must be > 0.")

    if max_utterance_seconds <= 0:
        raise ValueError("max_utterance_seconds must be > 0.")

    if min_utterance_seconds >= max_utterance_seconds:
        raise ValueError("min_utterance_seconds must be < max_utterance_seconds.")

    clean_segments = _clean_transcript_segments(transcript_segments)
    if not clean_segments:
        return []

    audio = AudioSegment.from_wav(str(input_path))
    audio_duration_sec = len(audio) / 1000.0

    nonsilent_ranges_ms = silence.detect_nonsilent(
        audio,
        min_silence_len=min_silence_len_ms,
        silence_thresh=silence_thresh_dbfs,
        seek_step=10,
    )

    utterance_groups = _build_utterance_groups(
        segments=clean_segments,
        min_utterance_seconds=min_utterance_seconds,
        max_utterance_seconds=max_utterance_seconds,
    )

    records: list[DatasetRecord] = []

    for utterance_index, group in enumerate(utterance_groups):
        text = _join_segment_text(group)
        if not text:
            continue

        start_sec = max(0.0, float(group[0].start))
        end_sec = min(audio_duration_sec, float(group[-1].end))

        start_sec, end_sec = _adjust_bounds_with_silence(
            start_sec=start_sec,
            end_sec=end_sec,
            nonsilent_ranges_ms=nonsilent_ranges_ms,
            keep_silence_ms=keep_silence_ms,
            audio_duration_sec=audio_duration_sec,
        )

        duration_sec = end_sec - start_sec

        if duration_sec < min_utterance_seconds:
            continue

        if duration_sec > max_utterance_seconds:
            split_records = _split_long_group_and_export(
                audio=audio,
                group=group,
                video_id=video_id,
                source_url=source_url,
                output_dir=output_dir,
                start_index=len(records),
                min_utterance_seconds=min_utterance_seconds,
                max_utterance_seconds=max_utterance_seconds,
                overwrite=overwrite,
                silence_thresh_dbfs=silence_thresh_dbfs,
            )
            records.extend(split_records)
            continue

        utt_id = f"{video_id}_{len(records):04d}"
        output_path = output_dir / f"{utt_id}.wav"

        clip = _export_audio_clip(
            audio=audio,
            output_path=output_path,
            start_sec=start_sec,
            end_sec=end_sec,
            overwrite=overwrite,
        )

        if clip is None:
            continue

        validation = validate_audio_segment(
            clip,
            path=output_path,
            min_duration_sec=min_utterance_seconds,
            max_duration_sec=max_utterance_seconds + 0.25,
            silence_dbfs_threshold=silence_thresh_dbfs,
        )

        if not validation.is_valid:
            continue

        records.append(
            DatasetRecord(
                utt_id=utt_id,
                video_id=video_id,
                source_url=source_url,
                audio_path=output_path,
                start_sec=start_sec,
                end_sec=end_sec,
                duration_sec=duration_sec,
                transcript=text,
            )
        )

    return records


def _clean_transcript_segments(
    transcript_segments: list[TranscriptSegment],
) -> list[TranscriptSegment]:
    clean_segments: list[TranscriptSegment] = []

    for segment in transcript_segments:
        text = " ".join(segment.text.strip().split())
        if not text:
            continue

        start = float(segment.start)
        end = float(segment.end)

        if end <= start:
            continue

        clean_segments.append(
            TranscriptSegment(
                start=start,
                end=end,
                text=text,
                avg_logprob=segment.avg_logprob,
                no_speech_prob=segment.no_speech_prob,
            )
        )

    clean_segments.sort(key=lambda item: (item.start, item.end))
    return clean_segments


def _build_utterance_groups(
    *,
    segments: list[TranscriptSegment],
    min_utterance_seconds: float,
    max_utterance_seconds: float,
) -> list[list[TranscriptSegment]]:
    groups: list[list[TranscriptSegment]] = []
    current: list[TranscriptSegment] = []

    for segment in segments:
        if not current:
            current = [segment]
            continue

        candidate = current + [segment]
        candidate_duration = float(candidate[-1].end) - float(candidate[0].start)
        current_duration = float(current[-1].end) - float(current[0].start)

        if candidate_duration > max_utterance_seconds:
            if current_duration >= min_utterance_seconds:
                groups.append(current)
                current = [segment]
            else:
                # If current is still too short, include candidate anyway;
                # long group will be handled later by fallback splitting.
                current = candidate
                groups.append(current)
                current = []
            continue

        current = candidate

        current_text = _join_segment_text(current)
        current_duration = float(current[-1].end) - float(current[0].start)

        if (
            current_duration >= min_utterance_seconds
            and _ends_with_sentence_boundary(current_text)
        ):
            groups.append(current)
            current = []

    if current:
        current_duration = float(current[-1].end) - float(current[0].start)
        if current_duration >= min_utterance_seconds:
            groups.append(current)
        elif groups:
            merged = groups[-1] + current
            merged_duration = float(merged[-1].end) - float(merged[0].start)
            if merged_duration <= max_utterance_seconds:
                groups[-1] = merged

    return groups


def _split_long_group_and_export(
    *,
    audio: AudioSegment,
    group: list[TranscriptSegment],
    video_id: str,
    source_url: str,
    output_dir: Path,
    start_index: int,
    min_utterance_seconds: float,
    max_utterance_seconds: float,
    overwrite: bool,
    silence_thresh_dbfs: float,
) -> list[DatasetRecord]:
    records: list[DatasetRecord] = []
    sub_groups = _build_duration_only_groups(
        group,
        min_utterance_seconds=min_utterance_seconds,
        max_utterance_seconds=max_utterance_seconds,
    )

    for sub_group in sub_groups:
        text = _join_segment_text(sub_group)
        if not text:
            continue

        start_sec = max(0.0, float(sub_group[0].start))
        end_sec = min(len(audio) / 1000.0, float(sub_group[-1].end))
        duration_sec = end_sec - start_sec

        if duration_sec < min_utterance_seconds or duration_sec > max_utterance_seconds:
            continue

        utt_id = f"{video_id}_{start_index + len(records):04d}"
        output_path = output_dir / f"{utt_id}.wav"

        clip = _export_audio_clip(
            audio=audio,
            output_path=output_path,
            start_sec=start_sec,
            end_sec=end_sec,
            overwrite=overwrite,
        )

        if clip is None:
            continue

        validation = validate_audio_segment(
            clip,
            path=output_path,
            min_duration_sec=min_utterance_seconds,
            max_duration_sec=max_utterance_seconds + 0.25,
            silence_dbfs_threshold=silence_thresh_dbfs,
        )

        if not validation.is_valid:
            continue

        records.append(
            DatasetRecord(
                utt_id=utt_id,
                video_id=video_id,
                source_url=source_url,
                audio_path=output_path,
                start_sec=start_sec,
                end_sec=end_sec,
                duration_sec=duration_sec,
                transcript=text,
            )
        )

    return records


def _build_duration_only_groups(
    segments: list[TranscriptSegment],
    *,
    min_utterance_seconds: float,
    max_utterance_seconds: float,
) -> list[list[TranscriptSegment]]:
    groups: list[list[TranscriptSegment]] = []
    current: list[TranscriptSegment] = []

    for segment in segments:
        if not current:
            current = [segment]
            continue

        candidate = current + [segment]
        candidate_duration = float(candidate[-1].end) - float(candidate[0].start)

        if candidate_duration <= max_utterance_seconds:
            current = candidate
            continue

        current_duration = float(current[-1].end) - float(current[0].start)
        if current_duration >= min_utterance_seconds:
            groups.append(current)
            current = [segment]
        else:
            current = []

    if current:
        current_duration = float(current[-1].end) - float(current[0].start)
        if min_utterance_seconds <= current_duration <= max_utterance_seconds:
            groups.append(current)

    return groups


def _adjust_bounds_with_silence(
    *,
    start_sec: float,
    end_sec: float,
    nonsilent_ranges_ms: list[list[int]],
    keep_silence_ms: int,
    audio_duration_sec: float,
) -> tuple[float, float]:
    if not nonsilent_ranges_ms:
        return start_sec, end_sec

    start_ms = int(round(start_sec * 1000))
    end_ms = int(round(end_sec * 1000))

    overlapping_ranges = [
        (range_start, range_end)
        for range_start, range_end in nonsilent_ranges_ms
        if range_start < end_ms and range_end > start_ms
    ]

    if not overlapping_ranges:
        return start_sec, end_sec

    adjusted_start_ms = max(0, min(range_start for range_start, _ in overlapping_ranges) - keep_silence_ms)
    adjusted_end_ms = min(
        int(round(audio_duration_sec * 1000)),
        max(range_end for _, range_end in overlapping_ranges) + keep_silence_ms,
    )

    if adjusted_end_ms <= adjusted_start_ms:
        return start_sec, end_sec

    return adjusted_start_ms / 1000.0, adjusted_end_ms / 1000.0


def _export_audio_clip(
    *,
    audio: AudioSegment,
    output_path: Path,
    start_sec: float,
    end_sec: float,
    overwrite: bool,
) -> AudioSegment | None:
    """Export a clip and return the in-memory AudioSegment, or None on failure.

    Returns the existing clip loaded from disk when reusing, or the freshly
    sliced segment when writing.  Callers use the returned object to validate
    without a redundant ffmpeg re-read.
    """
    start_ms = max(0, int(round(start_sec * 1000)))
    end_ms = min(len(audio), int(round(end_sec * 1000)))

    if end_ms <= start_ms:
        return None

    clip = audio[start_ms:end_ms]

    if output_path.exists() and not overwrite:
        return clip

    output_path.parent.mkdir(parents=True, exist_ok=True)
    clip.export(str(output_path), format="wav")
    return clip


def _join_segment_text(segments: list[TranscriptSegment]) -> str:
    return " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()


def _ends_with_sentence_boundary(text: str) -> bool:
    cleaned = text.strip()
    if not cleaned:
        return False
    return bool(SENTENCE_END_RE.search(cleaned))