from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from yt_dlp import YoutubeDL

from malayalam_tts_dataset.youtube.url_utils import canonical_watch_url, extract_video_id


@dataclass(frozen=True)
class DownloadResult:
    video_id: str
    source_url: str
    wav_path: Path
    title: str | None
    success: bool
    error: str | None = None
    used_existing: bool = False
    fresh_download: bool = False
    download_attempted: bool = False


@dataclass(frozen=True)
class YouTubeDownloadSettings:
    audio_full_dir: Path
    sample_rate: int = 16000
    channels: int = 1
    cookie_file: Path | None = None
    use_browser_cookies: bool = False
    browser_for_cookies: str = "chrome"
    sleep_interval: int = 8
    max_sleep_interval: int = 15
    retries: int = 3
    ratelimit: int | None = 1_000_000
    reuse_existing_audio: bool = True
    fetch_title_for_existing_audio: bool = False
    quiet: bool = True


class YouTubeAudioDownloader:
    """
    Downloads YouTube video audio as mono WAV using yt-dlp + FFmpeg.
    """

    def __init__(self, settings: YouTubeDownloadSettings) -> None:
        self.settings = settings
        self.settings.audio_full_dir.mkdir(parents=True, exist_ok=True)

    def download(self, url: str) -> DownloadResult:
        """
        Download one YouTube video's audio as WAV.

        Args:
            url:
                YouTube video URL.

        Returns:
            DownloadResult
        """
        try:
            video_id = extract_video_id(url)
            source_url = canonical_watch_url(url)
        except Exception as exc:
            return DownloadResult(
                video_id="",
                source_url=url,
                wav_path=Path(),
                title=None,
                success=False,
                error=str(exc),
                used_existing=False,
                fresh_download=False,
                download_attempted=False,
            )

        wav_path = self.settings.audio_full_dir / f"{video_id}.wav"

        if wav_path.exists() and self.settings.reuse_existing_audio:
            title = None
            if self.settings.fetch_title_for_existing_audio:
                title = self._try_fetch_title(source_url)
            return DownloadResult(
                video_id=video_id,
                source_url=source_url,
                wav_path=wav_path,
                title=title,
                success=True,
                error=None,
                used_existing=True,
                fresh_download=False,
                download_attempted=False,
            )

        opts = self._build_download_options()

        try:
            with YoutubeDL(opts) as ydl:
                info = ydl.extract_info(source_url, download=True)

            title = self._extract_title(info)

            if not wav_path.exists():
                possible_wavs = sorted(self.settings.audio_full_dir.glob(f"{video_id}*.wav"))
                if len(possible_wavs) == 1:
                    wav_path = possible_wavs[0]

            if not wav_path.exists():
                return DownloadResult(
                    video_id=video_id,
                    source_url=source_url,
                    wav_path=wav_path,
                    title=title,
                    success=False,
                    error=f"Download completed but WAV file not found: {wav_path}",
                    used_existing=False,
                    fresh_download=False,
                    download_attempted=True,
                )

            return DownloadResult(
                video_id=video_id,
                source_url=source_url,
                wav_path=wav_path,
                title=title,
                success=True,
                error=None,
                used_existing=False,
                fresh_download=True,
                download_attempted=True,
            )

        except Exception as exc:
            error_msg = str(exc)
            if wav_path.exists() and not self.settings.reuse_existing_audio:
                error_msg = f"Fresh download failed and reuse_existing_audio is false. yt-dlp error: {exc}"
            
            return DownloadResult(
                video_id=video_id,
                source_url=source_url,
                wav_path=wav_path,
                title=None,
                success=False,
                error=error_msg,
                used_existing=False,
                fresh_download=False,
                download_attempted=True,
            )

    def _build_download_options(self) -> dict[str, Any]:
        outtmpl = str(self.settings.audio_full_dir / "%(id)s.%(ext)s")

        opts: dict[str, Any] = {
            "format": "bestaudio/best",
            "outtmpl": outtmpl,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                    "preferredquality": "192",
                }
            ],
            "postprocessor_args": [
                "-ar",
                str(self.settings.sample_rate),
                "-ac",
                str(self.settings.channels),
            ],
            "quiet": self.settings.quiet,
            "noplaylist": True,
            "sleep_interval": self.settings.sleep_interval,
            "max_sleep_interval": self.settings.max_sleep_interval,
            "retries": self.settings.retries,
        }

        if self.settings.ratelimit is not None:
            opts["ratelimit"] = self.settings.ratelimit

        self._apply_cookies(opts)

        return opts

    def _build_metadata_options(self) -> dict[str, Any]:
        opts: dict[str, Any] = {
            "quiet": True,
            "skip_download": True,
            "noplaylist": True,
        }
        self._apply_cookies(opts)
        return opts

    def _apply_cookies(self, opts: dict[str, Any]) -> None:
        if self.settings.use_browser_cookies:
            opts["cookiesfrombrowser"] = (self.settings.browser_for_cookies,)

        if self.settings.cookie_file and self.settings.cookie_file.exists():
            opts["cookiefile"] = str(self.settings.cookie_file)

    def _try_fetch_title(self, source_url: str) -> str | None:
        try:
            with YoutubeDL(self._build_metadata_options()) as ydl:
                info = ydl.extract_info(source_url, download=False)
            return self._extract_title(info)
        except Exception:
            return None

    @staticmethod
    def _extract_title(info: Any) -> str | None:
        if isinstance(info, dict):
            title = info.get("title")
            if isinstance(title, str) and title.strip():
                return title.strip()
        return None