from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import whisper


@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    text: str
    avg_logprob: float | None = None
    no_speech_prob: float | None = None


_MALAYALAM_UNICODE_START = 0x0D00
_MALAYALAM_UNICODE_END = 0x0D7F
_MALAYALAM_MIN_SCRIPT_RATIO = 0.5


def _malayalam_script_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    malayalam_count = sum(
        1 for c in chars
        if _MALAYALAM_UNICODE_START <= ord(c) <= _MALAYALAM_UNICODE_END
    )
    return malayalam_count / len(chars)


class WhisperTranscriber:
    """
    Thin production-style wrapper around openai-whisper.

    The Whisper model is lazy-loaded once and reused for all files.
    """

    _SCRIPT_PRIMERS: dict[str, str] = {
        "ml": "ഇത് മലയാളം ഭാഷയിലുള്ള ഓഡിയോ ആണ്.",
    }

    _SCRIPT_FILTERS: dict[str, Any] = {
        "ml": _malayalam_script_ratio,
    }

    def __init__(
        self,
        model_name: str = "small",
        language: str = "ml",
        device: str | None = None,
        keep_empty_segments: bool = False,
        initial_prompt: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.language = language
        self.device = device or self._resolve_device()
        self.keep_empty_segments = keep_empty_segments
        self.initial_prompt = initial_prompt if initial_prompt is not None else self._SCRIPT_PRIMERS.get(language, "")
        self._script_ratio_fn = self._SCRIPT_FILTERS.get(language)
        self._model: Any | None = None

    @property
    def model(self) -> Any:
        """
        Lazy-load Whisper model.

        Returns:
            Loaded Whisper model.
        """
        if self._model is None:
            self._model = whisper.load_model(self.model_name, device=self.device)
        return self._model

    def transcribe(self, wav_path: str | Path) -> list[TranscriptSegment]:
        """
        Transcribe one audio file.

        Args:
            wav_path:
                Path to WAV/audio file.

        Returns:
            List of TranscriptSegment.
        """
        audio_path = Path(wav_path).expanduser().resolve()
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        result = self.model.transcribe(
            str(audio_path),
            language=self.language,
            fp16=self.device == "cuda",
            initial_prompt=self.initial_prompt or None,
            condition_on_previous_text=False,
            word_timestamps=True,
        )

        raw_segments = result.get("segments", [])
        if not isinstance(raw_segments, list):
            raise RuntimeError("Whisper result did not contain a valid segments list.")

        segments: list[TranscriptSegment] = []

        for raw_segment in raw_segments:
            if not isinstance(raw_segment, dict):
                continue

            text = str(raw_segment.get("text", "")).strip()

            if not text and not self.keep_empty_segments:
                continue

            if self._script_ratio_fn is not None:
                ratio = self._script_ratio_fn(text)
                if ratio < _MALAYALAM_MIN_SCRIPT_RATIO:
                    continue

            start = float(raw_segment.get("start", 0.0))
            end = float(raw_segment.get("end", start))

            if end < start:
                end = start

            avg_logprob = _safe_optional_float(raw_segment.get("avg_logprob"))
            no_speech_prob = _safe_optional_float(raw_segment.get("no_speech_prob"))

            segments.append(
                TranscriptSegment(
                    start=start,
                    end=end,
                    text=text,
                    avg_logprob=avg_logprob,
                    no_speech_prob=no_speech_prob,
                )
            )

        return segments

    @staticmethod
    def _resolve_device() -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"


def _safe_optional_float(value: Any) -> float | None:
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def transcript_segments_to_dicts(
    segments: list[TranscriptSegment],
) -> list[dict[str, Any]]:
    """
    Convert transcript segments to dictionaries.

    Args:
        segments:
            TranscriptSegment list.

    Returns:
        List of dictionaries.
    """
    return [asdict(segment) for segment in segments]


def save_transcript_json(
    *,
    video_id: str,
    source_path: str | Path,
    model_name: str,
    language: str,
    segments: list[TranscriptSegment],
    output_path: str | Path,
) -> Path:
    """
    Save transcription output to JSON.

    Output structure:
        {
          "video_id": "...",
          "source_path": "...",
          "model_name": "small",
          "language": "ml",
          "segments": [...]
        }

    Args:
        video_id:
            YouTube video ID.
        source_path:
            Transcribed audio path.
        model_name:
            Whisper model name.
        language:
            ASR language code.
        segments:
            Transcript segments.
        output_path:
            JSON output path.

    Returns:
        Resolved output path.
    """
    resolved_output_path = Path(output_path).expanduser().resolve()
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "video_id": video_id,
        "source_path": str(Path(source_path).expanduser().resolve()),
        "model_name": model_name,
        "language": language,
        "segments": transcript_segments_to_dicts(segments),
    }

    with resolved_output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return resolved_output_path