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
        return self._pipe

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

        result = self.model(
            str(audio_path),
            chunk_length_s=30,
            stride_length_s=5,
            return_timestamps=True,
            generate_kwargs={
                "language": self.language,
                "task": "transcribe",
            },
        )

        raw_chunks = result.get("chunks", [])
        if not isinstance(raw_chunks, list):
            raise RuntimeError("Pipeline result did not contain a valid chunks list.")

        segments: list[TranscriptSegment] = []

        for chunk in raw_chunks:
            if not isinstance(chunk, dict):
                continue

            text = str(chunk.get("text", "")).strip()

            if not text and not self.keep_empty_segments:
                continue

            if self._script_ratio_fn is not None:
                if self._script_ratio_fn(text) < _MALAYALAM_MIN_SCRIPT_RATIO:
                    continue

            timestamp = chunk.get("timestamp") or (0.0, 0.0)
            start = float(timestamp[0] or 0.0)
            end = float(timestamp[1] or start)

            if end < start:
                end = start

            segments.append(
                TranscriptSegment(
                    start=start,
                    end=end,
                    text=text,
                )
            )

        return segments

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
