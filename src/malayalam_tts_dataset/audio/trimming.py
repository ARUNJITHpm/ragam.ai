from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydub import AudioSegment

from malayalam_tts_dataset.audio.validation import validate_audio_file


@dataclass(frozen=True)
class TrimResult:
    input_path: Path
    output_path: Path
    start_sec: float
    end_sec: float
    duration_sec: float
    success: bool
    error: str | None = None


def trim_audio_window(
    input_wav_path: str | Path,
    output_wav_path: str | Path,
    start_seconds: float,
    duration_seconds: float,
    overwrite: bool = False,
) -> TrimResult:
    """
    Trim an exact audio window from a WAV file.

    Example:
        start_seconds=60
        duration_seconds=90

    Means:
        Keep 60s -> 150s.

    Args:
        input_wav_path:
            Source WAV path.
        output_wav_path:
            Destination trimmed WAV path.
        start_seconds:
            Start timestamp in seconds.
        duration_seconds:
            Duration to keep in seconds.
        overwrite:
            Whether to overwrite existing output.

    Returns:
        TrimResult.
    """
    input_path = Path(input_wav_path).expanduser().resolve()
    output_path = Path(output_wav_path).expanduser().resolve()

    start_sec = float(start_seconds)
    duration_sec = float(duration_seconds)
    end_sec = start_sec + duration_sec

    if start_sec < 0:
        return TrimResult(
            input_path=input_path,
            output_path=output_path,
            start_sec=start_sec,
            end_sec=end_sec,
            duration_sec=duration_sec,
            success=False,
            error="start_seconds must be >= 0.",
        )

    if duration_sec <= 0:
        return TrimResult(
            input_path=input_path,
            output_path=output_path,
            start_sec=start_sec,
            end_sec=end_sec,
            duration_sec=duration_sec,
            success=False,
            error="duration_seconds must be > 0.",
        )

    source_validation = validate_audio_file(
        input_path,
        min_duration_sec=end_sec,
    )

    if not source_validation.is_valid:
        return TrimResult(
            input_path=input_path,
            output_path=output_path,
            start_sec=start_sec,
            end_sec=end_sec,
            duration_sec=duration_sec,
            success=False,
            error="Source audio validation failed: "
            + "; ".join(source_validation.errors),
        )

    if output_path.exists() and not overwrite:
        existing_validation = validate_audio_file(
            output_path,
            min_duration_sec=max(0.1, duration_sec - 0.25),
        )

        if existing_validation.is_valid:
            return TrimResult(
                input_path=input_path,
                output_path=output_path,
                start_sec=start_sec,
                end_sec=end_sec,
                duration_sec=existing_validation.duration_sec or duration_sec,
                success=True,
                error=None,
            )

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        audio = AudioSegment.from_wav(str(input_path))

        start_ms = int(round(start_sec * 1000))
        end_ms = int(round(end_sec * 1000))

        trimmed = audio[start_ms:end_ms]

        if len(trimmed) <= 0:
            return TrimResult(
                input_path=input_path,
                output_path=output_path,
                start_sec=start_sec,
                end_sec=end_sec,
                duration_sec=0.0,
                success=False,
                error="Trimmed audio is empty.",
            )

        trimmed.export(str(output_path), format="wav")

        output_validation = validate_audio_file(
            output_path,
            min_duration_sec=max(0.1, duration_sec - 0.25),
        )

        if not output_validation.is_valid:
            return TrimResult(
                input_path=input_path,
                output_path=output_path,
                start_sec=start_sec,
                end_sec=end_sec,
                duration_sec=output_validation.duration_sec or 0.0,
                success=False,
                error="Trimmed audio validation failed: "
                + "; ".join(output_validation.errors),
            )

        return TrimResult(
            input_path=input_path,
            output_path=output_path,
            start_sec=start_sec,
            end_sec=end_sec,
            duration_sec=output_validation.duration_sec or duration_sec,
            success=True,
            error=None,
        )

    except Exception as exc:
        return TrimResult(
            input_path=input_path,
            output_path=output_path,
            start_sec=start_sec,
            end_sec=end_sec,
            duration_sec=duration_sec,
            success=False,
            error=str(exc),
        )


def trim_result_to_dict(result: TrimResult) -> dict:
    """
    Convert trim result to a JSON-serializable dictionary.

    Args:
        result:
            TrimResult.

    Returns:
        dict
    """
    return {
        "input_path": str(result.input_path),
        "output_path": str(result.output_path),
        "start_sec": result.start_sec,
        "end_sec": result.end_sec,
        "duration_sec": result.duration_sec,
        "success": result.success,
        "error": result.error,
    }