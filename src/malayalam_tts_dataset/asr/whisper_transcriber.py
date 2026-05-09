from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import pipeline as hf_pipeline


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

# Gap between words (seconds) that triggers a new segment
_WORD_GAP_SPLIT_SEC = 0.8

# Standard alignment heads per Whisper model size (decoder layer count → size).
# Fine-tuned models often omit these from their generation config; we patch them
# in so that return_timestamps="word" works correctly.
_DECODER_LAYERS_TO_SIZE = {4: "tiny", 6: "base", 12: "small", 24: "medium", 32: "large"}
_ALIGNMENT_HEADS: dict[str, list[list[int]]] = {
    "tiny":   [[2, 2], [3, 0], [3, 2], [3, 4], [3, 5], [3, 7],
               [3, 8], [3, 9], [3, 10], [3, 11], [3, 12], [3, 13]],
    "base":   [[3, 1], [4, 2], [4, 5], [4, 7], [5, 1], [5, 2], [5, 4], [5, 6]],
    "small":  [[1, 0], [2, 6], [2, 8], [3, 5], [3, 6], [4, 3],
               [4, 5], [4, 7], [5, 1], [5, 3], [5, 5]],
    "medium": [[11, 4], [14, 1], [14, 14], [15, 4],
               [13, 13], [13, 7], [13, 5], [14, 7]],
    "large":  [[9, 19], [11, 2], [11, 4], [12, 0], [12, 6],
               [13, 2], [13, 6], [14, 0], [14, 7], [15, 0], [15, 6]],
}


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
    Transcriber using HuggingFace transformers pipeline.
    Defaults to thennal/whisper-medium-ml, a Whisper medium model
    fine-tuned specifically on Malayalam speech.
    """

    _SCRIPT_FILTERS: dict[str, Any] = {
        "ml": _malayalam_script_ratio,
    }

    def __init__(
        self,
        model_name: str = "thennal/whisper-medium-ml",
        language: str = "ml",
        device: str | None = None,
        keep_empty_segments: bool = False,
        initial_prompt: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.language = language
        self.device = device or self._resolve_device()
        self.keep_empty_segments = keep_empty_segments
        self._script_ratio_fn = self._SCRIPT_FILTERS.get(language)
        self._pipe: Any | None = None

    @property
    def model(self) -> Any:
        if self._pipe is None:
            self._pipe = hf_pipeline(
                "automatic-speech-recognition",
                model=self.model_name,
                device=self.device,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
            self._patch_alignment_heads()
        return self._pipe

    def _patch_alignment_heads(self) -> None:
        """Inject alignment_heads if the model's generation config omits them.

        Fine-tuned Whisper models (e.g. thennal/whisper-medium-ml) are often
        saved without alignment_heads, which are required for word-level
        timestamps.  We detect the model size from decoder_layers and apply
        the standard OpenAI heads for that size.
        """
        gen_config = self._pipe.model.generation_config
        if getattr(gen_config, "alignment_heads", None):
            return

        num_layers = getattr(self._pipe.model.config, "decoder_layers", None)
        size = _DECODER_LAYERS_TO_SIZE.get(num_layers, "medium")
        gen_config.alignment_heads = _ALIGNMENT_HEADS[size]

    def transcribe(self, wav_path: str | Path) -> list[TranscriptSegment]:
        """
        Transcribe one audio file, returning utterance-level segments.

        Uses word-level timestamps from the pipeline and groups words into
        utterance segments by pause gaps and sentence-ending punctuation.
        This handles models (e.g. thennal/whisper-medium-ml) that return
        a single chunk with null timestamps in segment mode.

        Args:
            wav_path:
                Path to WAV/audio file.

        Returns:
            List of TranscriptSegment.
        """
        audio_path = Path(wav_path).expanduser().resolve()
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        result = self.model(
            str(audio_path),
            chunk_length_s=30,
            stride_length_s=5,
            return_timestamps="word",
            generate_kwargs={
                "language": self.language,
                "task": "transcribe",
                "num_beams": 1,       # greedy decoding — avoids MPS/GPU OOM on Apple Silicon
            },
        )

        word_chunks = result.get("chunks", [])
        if not isinstance(word_chunks, list):
            raise RuntimeError("Pipeline result did not contain a valid chunks list.")

        segments = self._group_words_into_segments(word_chunks)
        return segments

    def _group_words_into_segments(
        self, word_chunks: list[dict[str, Any]]
    ) -> list[TranscriptSegment]:
        """
        Group word-level chunks into utterance segments.

        Splits on:
        - Pause gap between consecutive words >= _WORD_GAP_SPLIT_SEC
        - Sentence-ending punctuation (. ! ? । ॥)
        """
        segments: list[TranscriptSegment] = []
        current_words: list[str] = []
        current_start: float | None = None
        current_end: float = 0.0
        prev_end: float = 0.0

        sentence_endings = {".", "!", "?", "।", "॥"}

        for chunk in word_chunks:
            if not isinstance(chunk, dict):
                continue

            text = str(chunk.get("text", "")).strip()
            if not text:
                continue

            ts = chunk.get("timestamp") or (None, None)
            word_start = float(ts[0]) if ts[0] is not None else prev_end
            word_end = float(ts[1]) if ts[1] is not None else word_start

            gap = word_start - prev_end if current_words else 0.0
            ends_sentence = any(text.endswith(p) for p in sentence_endings)
            split_on_gap = gap >= _WORD_GAP_SPLIT_SEC and current_words

            if split_on_gap:
                seg = self._flush_segment(current_words, current_start, current_end)
                if seg:
                    segments.append(seg)
                current_words = []
                current_start = None

            if current_start is None:
                current_start = word_start

            current_words.append(text)
            current_end = word_end
            prev_end = word_end

            if ends_sentence and current_words:
                seg = self._flush_segment(current_words, current_start, current_end)
                if seg:
                    segments.append(seg)
                current_words = []
                current_start = None

        if current_words:
            seg = self._flush_segment(current_words, current_start, current_end)
            if seg:
                segments.append(seg)

        if self._script_ratio_fn is not None:
            segments = [
                s for s in segments
                if self._script_ratio_fn(s.text) >= _MALAYALAM_MIN_SCRIPT_RATIO
                or self.keep_empty_segments
            ]

        return segments

    def _flush_segment(
        self,
        words: list[str],
        start: float | None,
        end: float,
    ) -> TranscriptSegment | None:
        text = " ".join(words).strip()
        if not text and not self.keep_empty_segments:
            return None
        seg_start = start if start is not None else 0.0
        seg_end = max(end, seg_start)
        return TranscriptSegment(start=seg_start, end=seg_end, text=text)

    @staticmethod
    def _resolve_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"


def transcript_segments_to_dicts(
    segments: list[TranscriptSegment],
) -> list[dict[str, Any]]:
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
