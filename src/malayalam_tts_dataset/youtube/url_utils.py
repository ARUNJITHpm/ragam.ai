from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse


class YouTubeURLError(ValueError):
    """Raised when a YouTube URL is invalid or unsupported."""


YOUTUBE_VIDEO_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{11}$")


def extract_video_id(url: str) -> str:
    """
    Extract a stable YouTube video ID from a supported YouTube URL.

    Supported examples:
        https://www.youtube.com/watch?v=VIDEO_ID
        https://www.youtube.com/watch?v=VIDEO_ID&t=10s
        https://youtu.be/VIDEO_ID
        https://youtu.be/VIDEO_ID?t=10
        https://m.youtube.com/watch?v=VIDEO_ID

    Args:
        url:
            YouTube video URL.

    Returns:
        11-character YouTube video ID.

    Raises:
        YouTubeURLError:
            If URL is invalid or video ID cannot be extracted.
    """
    if not isinstance(url, str) or not url.strip():
        raise YouTubeURLError("YouTube URL must be a non-empty string.")

    clean_url = url.strip()
    parsed = urlparse(clean_url)

    if parsed.scheme not in {"http", "https"}:
        raise YouTubeURLError(
            f"Unsupported URL scheme '{parsed.scheme}'. Expected http or https."
        )

    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]

    video_id: str | None = None

    if host in {"youtube.com", "m.youtube.com", "music.youtube.com"}:
        if parsed.path == "/watch":
            query = parse_qs(parsed.query)
            values = query.get("v")
            if values:
                video_id = values[0]
        elif parsed.path.startswith("/shorts/"):
            video_id = parsed.path.split("/shorts/", 1)[1].split("/", 1)[0]
        elif parsed.path.startswith("/embed/"):
            video_id = parsed.path.split("/embed/", 1)[1].split("/", 1)[0]
        else:
            raise YouTubeURLError(
                f"Unsupported YouTube URL path '{parsed.path}'. "
                "Expected /watch, /shorts/<id>, or /embed/<id>."
            )

    elif host == "youtu.be":
        video_id = parsed.path.lstrip("/").split("/", 1)[0]

    else:
        raise YouTubeURLError(
            f"Unsupported YouTube host '{parsed.netloc}'. "
            "Expected youtube.com, m.youtube.com, music.youtube.com, or youtu.be."
        )

    if not video_id:
        raise YouTubeURLError(f"Could not extract video ID from URL: {url}")

    video_id = video_id.strip()

    if not is_valid_video_id(video_id):
        raise YouTubeURLError(
            f"Invalid YouTube video ID '{video_id}'. "
            "Expected 11 characters using letters, numbers, '-' or '_'."
        )

    return video_id


def is_valid_video_id(video_id: str) -> bool:
    """
    Validate a YouTube video ID.

    Args:
        video_id:
            Candidate YouTube video ID.

    Returns:
        True if valid, else False.
    """
    if not isinstance(video_id, str):
        return False
    return bool(YOUTUBE_VIDEO_ID_PATTERN.fullmatch(video_id.strip()))


def canonical_watch_url(url: str) -> str:
    """
    Convert a supported YouTube URL to canonical watch URL.

    Args:
        url:
            YouTube URL.

    Returns:
        https://www.youtube.com/watch?v=<video_id>
    """
    video_id = extract_video_id(url)
    return f"https://www.youtube.com/watch?v={video_id}"