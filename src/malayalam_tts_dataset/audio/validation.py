from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydub import AudioSegment


@dataclass(frozen=True)
class AudioValidationResult:
    path: Path
    exists: bool
    duration_sec: float | None
    sample_rate: int | None
    channels: int | None
    rms_dbfs: float | None
    is_silent: bool
    is_too_short: bool
    is_valid: bool
    warnings: list[str]
    errors: list[str]


def validate_audio_file(
    path: str | Path,
    min_duration_sec: float | None = None,
    max_duration_sec: float | None = None,
    silence_dbfs_threshold: float = -55.0,
) -> AudioValidationResult:
    """
    Validate an audio file for dataset processing.

    Checks:
    - file exists
    - file is readable by pydub/ffmpeg
    - duration is readable
    - duration >= min_duration_sec if provided
    - duration <= max_duration_sec if provided
    - sample rate is readable
    - channel count is readable
    - audio is not fully/mostly silent

    Args:
        path:
            Audio path.
        min_duration_sec:
            Optional minimum required duration in seconds.
        max_duration_sec:
            Optional maximum allowed duration in seconds.
        silence_dbfs_threshold:
            Audio quieter than this dBFS is treated as silent.

    Returns:
        AudioValidationResult
    """
    audio_path = Path(path).expanduser().resolve()
    warnings: list[str] = []
    errors: list[str] = []

    exists = audio_path.exists()
    duration_sec: float | None = None
    sample_rate: int | None = None
    channels: int | None = None
    rms_dbfs: float | None = None
    is_silent = False
    is_too_short = False

    if not exists:
        errors.append(f"Audio file does not exist: {audio_path}")
        return AudioValidationResult(
            path=audio_path,
            exists=False,
            duration_sec=None,
            sample_rate=None,
            channels=None,
            rms_dbfs=None,
            is_silent=False,
            is_too_short=False,
            is_valid=False,
            warnings=warnings,
            errors=errors,
        )

    if not audio_path.is_file():
        errors.append(f"Audio path is not a file: {audio_path}")
        return AudioValidationResult(
            path=audio_path,
            exists=True,
            duration_sec=None,
            sample_rate=None,
            channels=None,
            rms_dbfs=None,
            is_silent=False,
            is_too_short=False,
            is_valid=False,
            warnings=warnings,
            errors=errors,
        )

    try:
        audio = AudioSegment.from_file(str(audio_path))
    except Exception as exc:
        errors.append(f"Audio file is not readable: {audio_path}. Error: {exc}")
        return AudioValidationResult(
            path=audio_path,
            exists=True,
            duration_sec=None,
            sample_rate=None,
            channels=None,
            rms_dbfs=None,
            is_silent=False,
            is_too_short=False,
            is_valid=False,
            warnings=warnings,
            errors=errors,
        )

    duration_sec = len(audio) / 1000.0
    sample_rate = int(audio.frame_rate)
    channels = int(audio.channels)
    rms_dbfs = float(audio.dBFS) if audio.dBFS != float("-inf") else float("-inf")

    if duration_sec <= 0:
        errors.append("Audio duration is zero.")
    elif duration_sec < 0.5:
        warnings.append(f"Audio is very short: {duration_sec:.3f}s")

    if min_duration_sec is not None:
        if min_duration_sec < 0:
            errors.append("min_duration_sec cannot be negative.")
        elif duration_sec < min_duration_sec:
            is_too_short = True
            errors.append(
                f"Audio is too short: {duration_sec:.3f}s. "
                f"Required at least {min_duration_sec:.3f}s."
            )

    if max_duration_sec is not None:
        if max_duration_sec <= 0:
            errors.append("max_duration_sec must be positive.")
        elif duration_sec > max_duration_sec:
            errors.append(
                f"Audio is too long: {duration_sec:.3f}s. "
                f"Maximum allowed is {max_duration_sec:.3f}s."
            )

    if sample_rate <= 0:
        errors.append("Invalid sample rate.")
    elif sample_rate not in {8000, 16000, 22050, 24000, 44100, 48000}:
        warnings.append(f"Unusual sample rate: {sample_rate}")

    if channels <= 0:
        errors.append("Invalid channel count.")
    elif channels > 2:
        warnings.append(f"Unexpected channel count: {channels}")

    if audio.rms == 0 or audio.dBFS == float("-inf"):
        is_silent = True
        errors.append("Audio is fully silent.")
    elif rms_dbfs is not None and rms_dbfs < silence_dbfs_threshold:
        is_silent = True
        errors.append(
            f"Audio is below silence threshold: {rms_dbfs:.2f} dBFS "
            f"< {silence_dbfs_threshold:.2f} dBFS."
        )

    is_valid = len(errors) == 0

    return AudioValidationResult(
        path=audio_path,
        exists=exists,
        duration_sec=duration_sec,
        sample_rate=sample_rate,
        channels=channels,
        rms_dbfs=rms_dbfs,
        is_silent=is_silent,
        is_too_short=is_too_short,
        is_valid=is_valid,
        warnings=warnings,
        errors=errors,
    )


def validation_result_to_dict(result: AudioValidationResult) -> dict:
    """
    Convert validation result to a JSON-serializable dictionary.

    Args:
        result:
            AudioValidationResult.

    Returns:
        dict
    """
    return {
        "path": str(result.path),
        "exists": result.exists,
        "duration_sec": result.duration_sec,
        "sample_rate": result.sample_rate,
        "channels": result.channels,
        "rms_dbfs": result.rms_dbfs,
        "is_silent": result.is_silent,
        "is_too_short": result.is_too_short,
        "is_valid": result.is_valid,
        "warnings": result.warnings,
        "errors": result.errors,
    }