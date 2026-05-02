from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from malayalam_tts_dataset.dataset.records import (
    DATASET_CSV_COLUMNS,
    DatasetRecord,
    dataset_record_to_dict,
)


def write_dataset_csv(records: list[DatasetRecord], output_path: str | Path) -> Path:
    resolved_path = Path(output_path).expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    with resolved_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DATASET_CSV_COLUMNS)
        writer.writeheader()

        for record in records:
            writer.writerow(dataset_record_to_dict(record))

    return resolved_path


def write_metadata_csv(records: list[DatasetRecord], output_path: str | Path) -> Path:
    resolved_path = Path(output_path).expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    with resolved_path.open("w", encoding="utf-8", newline="") as f:
        for record in records:
            audio_path = str(record.audio_path)
            transcript = _clean_metadata_text(record.transcript)
            f.write(f"{audio_path}|{transcript}\n")

    return resolved_path


def write_run_summary(summary: dict[str, Any], output_path: str | Path) -> Path:
    resolved_path = Path(output_path).expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    with resolved_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return resolved_path


def _clean_metadata_text(text: str) -> str:
    return " ".join(text.replace("|", " ").split())