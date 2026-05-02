from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetRecord:
    utt_id: str
    video_id: str
    source_url: str
    audio_path: Path
    start_sec: float
    end_sec: float
    duration_sec: float
    transcript: str


DATASET_CSV_COLUMNS = [
    "utt_id",
    "video_id",
    "source_url",
    "audio_path",
    "start_sec",
    "end_sec",
    "duration_sec",
    "transcript",
]


def dataset_record_to_dict(record: DatasetRecord) -> dict[str, Any]:
    data = asdict(record)
    data["audio_path"] = str(record.audio_path)
    data["start_sec"] = round(float(record.start_sec), 3)
    data["end_sec"] = round(float(record.end_sec), 3)
    data["duration_sec"] = round(float(record.duration_sec), 3)
    return data


def dataset_records_to_dicts(records: list[DatasetRecord]) -> list[dict[str, Any]]:
    return [dataset_record_to_dict(record) for record in records]