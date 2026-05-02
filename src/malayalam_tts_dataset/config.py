from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


class ConfigError(ValueError):
    """Raised when project configuration is invalid."""


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int
    channels: int
    trim_start_seconds: float
    trim_duration_seconds: float
    min_utterance_seconds: float
    max_utterance_seconds: float
    silence_thresh_dbfs: float
    min_silence_len_ms: int
    keep_silence_ms: int


@dataclass(frozen=True)
class ASRConfig:
    whisper_model: str
    language: str
    keep_empty_segments: bool


@dataclass(frozen=True)
class YouTubeConfig:
    cookie_file: Path
    use_browser_cookies: bool
    browser_for_cookies: str
    sleep_between_downloads: int
    ydl_sleep_interval: int
    ydl_max_sleep_interval: int
    retries: int
    ratelimit: int | None
    fetch_title_for_existing_audio: bool


@dataclass(frozen=True)
class SplitConfig:
    train_ratio: float
    val_ratio: float
    test_ratio: float
    random_seed: int


@dataclass(frozen=True)
class RuntimeConfig:
    overwrite_outputs: bool
    reuse_existing_audio: bool


@dataclass(frozen=True)
class AppConfig:
    video_urls: list[str]
    output_dir: Path
    audio: AudioConfig
    asr: ASRConfig
    youtube: YouTubeConfig
    split: SplitConfig
    runtime: RuntimeConfig
    project_root: Path

    @property
    def audio_full_dir(self) -> Path:
        return self.output_dir / "audio_full"

    @property
    def audio_trimmed_dir(self) -> Path:
        return self.output_dir / "audio_trimmed"

    @property
    def segments_dir(self) -> Path:
        return self.output_dir / "segments"

    @property
    def transcripts_dir(self) -> Path:
        return self.output_dir / "transcripts"

    @property
    def dataset_csv_path(self) -> Path:
        return self.output_dir / "dataset.csv"

    @property
    def metadata_csv_path(self) -> Path:
        return self.output_dir / "metadata.csv"

    @property
    def train_csv_path(self) -> Path:
        return self.output_dir / "train.csv"

    @property
    def val_csv_path(self) -> Path:
        return self.output_dir / "val.csv"

    @property
    def test_csv_path(self) -> Path:
        return self.output_dir / "test.csv"

    @property
    def train_metadata_csv_path(self) -> Path:
        return self.output_dir / "train_metadata.csv"

    @property
    def val_metadata_csv_path(self) -> Path:
        return self.output_dir / "val_metadata.csv"

    @property
    def test_metadata_csv_path(self) -> Path:
        return self.output_dir / "test_metadata.csv"

    @property
    def run_summary_path(self) -> Path:
        return self.output_dir / "run_summary.json"


def load_config(config_path: str | Path) -> AppConfig:
    load_dotenv()

    resolved_config_path = Path(config_path).expanduser().resolve()
    if not resolved_config_path.exists():
        raise ConfigError(f"Config file not found: {resolved_config_path}")

    project_root = _find_project_root(resolved_config_path)

    with resolved_config_path.open("r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    if not isinstance(raw_config, dict):
        raise ConfigError("Config file must contain a YAML mapping/object.")

    config = _parse_config(raw_config, project_root=project_root)
    validate_config(config)
    create_output_dirs(config)

    return config


def _find_project_root(config_path: Path) -> Path:
    if config_path.parent.name == "configs":
        return config_path.parent.parent.resolve()
    return Path.cwd().resolve()


def _parse_config(raw: dict[str, Any], project_root: Path) -> AppConfig:
    video_urls = raw.get("video_urls")
    if not isinstance(video_urls, list):
        raise ConfigError("'video_urls' must be a list of YouTube URLs.")

    output_dir_raw = raw.get("output_dir", "outputs/dataset_youtube")
    output_dir = _resolve_project_path(output_dir_raw, project_root)

    audio_raw = raw.get("audio", {})
    asr_raw = raw.get("asr", {})
    youtube_raw = raw.get("youtube", {})
    split_raw = raw.get("split", {})
    runtime_raw = raw.get("runtime", {})

    if not isinstance(audio_raw, dict):
        raise ConfigError("'audio' section must be a mapping.")
    if not isinstance(asr_raw, dict):
        raise ConfigError("'asr' section must be a mapping.")
    if not isinstance(youtube_raw, dict):
        raise ConfigError("'youtube' section must be a mapping.")
    if not isinstance(split_raw, dict):
        raise ConfigError("'split' section must be a mapping.")
    if not isinstance(runtime_raw, dict):
        raise ConfigError("'runtime' section must be a mapping.")

    cookie_file = _resolve_project_path(
        youtube_raw.get("cookie_file", "cookies/cookies.txt"),
        project_root,
    )

    audio = AudioConfig(
        sample_rate=int(audio_raw.get("sample_rate", 16000)),
        channels=int(audio_raw.get("channels", 1)),
        trim_start_seconds=float(audio_raw.get("trim_start_seconds", 60)),
        trim_duration_seconds=float(audio_raw.get("trim_duration_seconds", 90)),
        min_utterance_seconds=float(audio_raw.get("min_utterance_seconds", 2.0)),
        max_utterance_seconds=float(audio_raw.get("max_utterance_seconds", 12.0)),
        silence_thresh_dbfs=float(audio_raw.get("silence_thresh_dbfs", -40.0)),
        min_silence_len_ms=int(audio_raw.get("min_silence_len_ms", 350)),
        keep_silence_ms=int(audio_raw.get("keep_silence_ms", 120)),
    )

    asr = ASRConfig(
        whisper_model=str(asr_raw.get("whisper_model", "small")),
        language=str(asr_raw.get("language", "ml")),
        keep_empty_segments=bool(asr_raw.get("keep_empty_segments", False)),
    )

    youtube = YouTubeConfig(
        cookie_file=cookie_file,
        use_browser_cookies=bool(youtube_raw.get("use_browser_cookies", False)),
        browser_for_cookies=str(youtube_raw.get("browser_for_cookies", "chrome")),
        sleep_between_downloads=int(youtube_raw.get("sleep_between_downloads", 15)),
        ydl_sleep_interval=int(youtube_raw.get("ydl_sleep_interval", 8)),
        ydl_max_sleep_interval=int(youtube_raw.get("ydl_max_sleep_interval", 15)),
        retries=int(youtube_raw.get("retries", 3)),
        ratelimit=(
            int(youtube_raw["ratelimit"])
            if youtube_raw.get("ratelimit") is not None
            else None
        ),
        fetch_title_for_existing_audio=bool(youtube_raw.get("fetch_title_for_existing_audio", False)),
    )

    split = SplitConfig(
        train_ratio=float(split_raw.get("train_ratio", 0.9)),
        val_ratio=float(split_raw.get("val_ratio", 0.05)),
        test_ratio=float(split_raw.get("test_ratio", 0.05)),
        random_seed=int(split_raw.get("random_seed", 42)),
    )

    runtime = RuntimeConfig(
        overwrite_outputs=bool(runtime_raw.get("overwrite_outputs", False)),
        reuse_existing_audio=bool(runtime_raw.get("reuse_existing_audio", True)),
    )

    return AppConfig(
        video_urls=[str(url).strip() for url in video_urls],
        output_dir=output_dir,
        audio=audio,
        asr=asr,
        youtube=youtube,
        split=split,
        runtime=runtime,
        project_root=project_root,
    )


def _resolve_project_path(path_value: str | Path, project_root: Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()


def validate_config(config: AppConfig) -> None:
    if not config.video_urls:
        raise ConfigError("'video_urls' must contain at least one URL.")

    invalid_urls = [
        url for url in config.video_urls
        if not isinstance(url, str) or not url.strip()
    ]
    if invalid_urls:
        raise ConfigError(f"Invalid empty URL entries found: {invalid_urls}")

    if config.audio.sample_rate <= 0:
        raise ConfigError("'audio.sample_rate' must be positive.")

    if config.audio.channels not in {1, 2}:
        raise ConfigError("'audio.channels' must be 1 or 2.")

    if config.audio.trim_start_seconds < 0:
        raise ConfigError("'audio.trim_start_seconds' must be >= 0.")

    if config.audio.trim_duration_seconds <= 0:
        raise ConfigError("'audio.trim_duration_seconds' must be > 0.")

    if config.audio.min_utterance_seconds <= 0:
        raise ConfigError("'audio.min_utterance_seconds' must be > 0.")

    if config.audio.max_utterance_seconds <= 0:
        raise ConfigError("'audio.max_utterance_seconds' must be > 0.")

    if config.audio.min_utterance_seconds >= config.audio.max_utterance_seconds:
        raise ConfigError(
            "'audio.min_utterance_seconds' must be less than "
            "'audio.max_utterance_seconds'."
        )

    if config.audio.silence_thresh_dbfs >= 0:
        raise ConfigError("'audio.silence_thresh_dbfs' should be negative.")

    if config.audio.min_silence_len_ms <= 0:
        raise ConfigError("'audio.min_silence_len_ms' must be > 0.")

    if config.audio.keep_silence_ms < 0:
        raise ConfigError("'audio.keep_silence_ms' must be >= 0.")

    if not config.asr.whisper_model.strip():
        raise ConfigError("'asr.whisper_model' must be non-empty.")

    if not config.asr.language.strip():
        raise ConfigError("'asr.language' must be non-empty.")

    if config.youtube.use_browser_cookies and not config.youtube.browser_for_cookies:
        raise ConfigError(
            "'youtube.browser_for_cookies' must be set when "
            "'youtube.use_browser_cookies' is true."
        )

    if config.youtube.sleep_between_downloads < 0:
        raise ConfigError("'youtube.sleep_between_downloads' must be >= 0.")

    if config.youtube.ydl_sleep_interval < 0:
        raise ConfigError("'youtube.ydl_sleep_interval' must be >= 0.")

    if config.youtube.ydl_max_sleep_interval < config.youtube.ydl_sleep_interval:
        raise ConfigError(
            "'youtube.ydl_max_sleep_interval' must be >= "
            "'youtube.ydl_sleep_interval'."
        )

    if config.youtube.retries < 0:
        raise ConfigError("'youtube.retries' must be >= 0.")

    if config.youtube.ratelimit is not None and config.youtube.ratelimit <= 0:
        raise ConfigError("'youtube.ratelimit' must be positive or null.")

    ratio_sum = (
        config.split.train_ratio
        + config.split.val_ratio
        + config.split.test_ratio
    )

    if config.split.train_ratio <= 0:
        raise ConfigError("'split.train_ratio' must be > 0.")

    if config.split.val_ratio < 0:
        raise ConfigError("'split.val_ratio' must be >= 0.")

    if config.split.test_ratio < 0:
        raise ConfigError("'split.test_ratio' must be >= 0.")

    if abs(ratio_sum - 1.0) > 1e-6:
        raise ConfigError(
            "'split.train_ratio' + 'split.val_ratio' + 'split.test_ratio' "
            f"must equal 1.0. Got {ratio_sum}."
        )


def create_output_dirs(config: AppConfig) -> None:
    directories = [
        config.output_dir,
        config.audio_full_dir,
        config.audio_trimmed_dir,
        config.segments_dir,
        config.transcripts_dir,
        config.youtube.cookie_file.parent,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def config_to_dict(config: AppConfig) -> dict[str, Any]:
    return {
        "project_root": str(config.project_root),
        "video_urls": config.video_urls,
        "output_dir": str(config.output_dir),
        "paths": {
            "audio_full_dir": str(config.audio_full_dir),
            "audio_trimmed_dir": str(config.audio_trimmed_dir),
            "segments_dir": str(config.segments_dir),
            "transcripts_dir": str(config.transcripts_dir),
            "dataset_csv_path": str(config.dataset_csv_path),
            "metadata_csv_path": str(config.metadata_csv_path),
            "train_csv_path": str(config.train_csv_path),
            "val_csv_path": str(config.val_csv_path),
            "test_csv_path": str(config.test_csv_path),
            "train_metadata_csv_path": str(config.train_metadata_csv_path),
            "val_metadata_csv_path": str(config.val_metadata_csv_path),
            "test_metadata_csv_path": str(config.test_metadata_csv_path),
            "run_summary_path": str(config.run_summary_path),
        },
        "audio": {
            "sample_rate": config.audio.sample_rate,
            "channels": config.audio.channels,
            "trim_start_seconds": config.audio.trim_start_seconds,
            "trim_duration_seconds": config.audio.trim_duration_seconds,
            "min_utterance_seconds": config.audio.min_utterance_seconds,
            "max_utterance_seconds": config.audio.max_utterance_seconds,
            "silence_thresh_dbfs": config.audio.silence_thresh_dbfs,
            "min_silence_len_ms": config.audio.min_silence_len_ms,
            "keep_silence_ms": config.audio.keep_silence_ms,
        },
        "asr": {
            "whisper_model": config.asr.whisper_model,
            "language": config.asr.language,
            "keep_empty_segments": config.asr.keep_empty_segments,
        },
        "youtube": {
            "cookie_file": str(config.youtube.cookie_file),
            "use_browser_cookies": config.youtube.use_browser_cookies,
            "browser_for_cookies": config.youtube.browser_for_cookies,
            "sleep_between_downloads": config.youtube.sleep_between_downloads,
            "ydl_sleep_interval": config.youtube.ydl_sleep_interval,
            "ydl_max_sleep_interval": config.youtube.ydl_max_sleep_interval,
            "retries": config.youtube.retries,
            "ratelimit": config.youtube.ratelimit,
            "fetch_title_for_existing_audio": config.youtube.fetch_title_for_existing_audio,
        },
        "split": {
            "train_ratio": config.split.train_ratio,
            "val_ratio": config.split.val_ratio,
            "test_ratio": config.split.test_ratio,
            "random_seed": config.split.random_seed,
        },
        "runtime": {
            "overwrite_outputs": config.runtime.overwrite_outputs,
            "reuse_existing_audio": config.runtime.reuse_existing_audio,
        },
    }