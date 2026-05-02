from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

from malayalam_tts_dataset.dataset.records import DatasetRecord


@dataclass(frozen=True)
class DatasetSplit:
    train: list[DatasetRecord]
    val: list[DatasetRecord]
    test: list[DatasetRecord]
    strategy: str
    warnings: list[str]


def split_records(
    records: list[DatasetRecord],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
) -> DatasetSplit:
    """
    Split dataset records into train/val/test.

    Preferred strategy:
        Split by video_id to reduce leakage.

    Fallback:
        If only 1-2 videos exist, split by utterance with warning.

    Args:
        records:
            Dataset records.
        train_ratio:
            Train split ratio.
        val_ratio:
            Validation split ratio.
        test_ratio:
            Test split ratio.
        seed:
            Random seed.

    Returns:
        DatasetSplit.
    """
    warnings: list[str] = []

    if not records:
        return DatasetSplit(
            train=[],
            val=[],
            test=[],
            strategy="empty",
            warnings=["No records available to split."],
        )

    _validate_ratios(train_ratio, val_ratio, test_ratio)

    records_by_video: dict[str, list[DatasetRecord]] = defaultdict(list)
    for record in records:
        records_by_video[record.video_id].append(record)

    unique_video_ids = sorted(records_by_video.keys())

    if len(unique_video_ids) >= 3:
        return _split_by_video_id(
            records_by_video=records_by_video,
            video_ids=unique_video_ids,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )

    warnings.append(
        "Only 1-2 unique video_id values found. "
        "Falling back to utterance-level split. "
        "This may cause leakage between train/val/test."
    )

    return _split_by_utterance(
        records=records,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        warnings=warnings,
    )


def _validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    if train_ratio <= 0:
        raise ValueError("train_ratio must be > 0.")

    if val_ratio < 0:
        raise ValueError("val_ratio must be >= 0.")

    if test_ratio < 0:
        raise ValueError("test_ratio must be >= 0.")

    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0. Got {total}.")


def _split_by_video_id(
    *,
    records_by_video: dict[str, list[DatasetRecord]],
    video_ids: list[str],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> DatasetSplit:
    rng = random.Random(seed)
    shuffled_video_ids = video_ids[:]
    rng.shuffle(shuffled_video_ids)

    total_videos = len(shuffled_video_ids)

    train_count = max(1, int(round(total_videos * train_ratio)))
    val_count = int(round(total_videos * val_ratio))

    if train_count >= total_videos:
        train_count = total_videos - 2
        val_count = 1

    remaining_after_train = total_videos - train_count

    if val_count >= remaining_after_train:
        val_count = max(1, remaining_after_train - 1)

    train_ids = set(shuffled_video_ids[:train_count])
    val_ids = set(shuffled_video_ids[train_count:train_count + val_count])
    test_ids = set(shuffled_video_ids[train_count + val_count:])

    train_records = _collect_records(records_by_video, train_ids)
    val_records = _collect_records(records_by_video, val_ids)
    test_records = _collect_records(records_by_video, test_ids)

    return DatasetSplit(
        train=train_records,
        val=val_records,
        test=test_records,
        strategy="video_id",
        warnings=[],
    )


def _split_by_utterance(
    *,
    records: list[DatasetRecord],
    train_ratio: float,
    val_ratio: float,
    seed: int,
    warnings: list[str],
) -> DatasetSplit:
    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)

    total = len(shuffled)

    if total == 1:
        warnings.append("Only one utterance found. Assigning it to train split.")
        return DatasetSplit(
            train=shuffled,
            val=[],
            test=[],
            strategy="utterance",
            warnings=warnings,
        )

    train_count = max(1, int(round(total * train_ratio)))
    val_count = int(round(total * val_ratio))

    if train_count >= total:
        train_count = total - 1
        val_count = 0

    remaining_after_train = total - train_count

    if val_count >= remaining_after_train:
        val_count = max(0, remaining_after_train - 1)

    train = shuffled[:train_count]
    val = shuffled[train_count:train_count + val_count]
    test = shuffled[train_count + val_count:]

    return DatasetSplit(
        train=train,
        val=val,
        test=test,
        strategy="utterance",
        warnings=warnings,
    )


def _collect_records(
    records_by_video: dict[str, list[DatasetRecord]],
    video_ids: set[str],
) -> list[DatasetRecord]:
    records: list[DatasetRecord] = []
    for video_id in sorted(video_ids):
        records.extend(records_by_video[video_id])
    return records