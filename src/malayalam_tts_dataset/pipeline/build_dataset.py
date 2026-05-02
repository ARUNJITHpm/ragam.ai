from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from malayalam_tts_dataset.asr.whisper_transcriber import (
    WhisperTranscriber,
    save_transcript_json,
    transcript_segments_to_dicts,
)
from malayalam_tts_dataset.audio.segmentation import segment_transcribed_audio
from malayalam_tts_dataset.audio.trimming import trim_audio_window, trim_result_to_dict
from malayalam_tts_dataset.audio.validation import (
    validate_audio_file,
    validation_result_to_dict,
)
from malayalam_tts_dataset.config import AppConfig, load_config
from malayalam_tts_dataset.dataset.records import (
    DatasetRecord,
    dataset_records_to_dicts,
)
from malayalam_tts_dataset.dataset.splitter import DatasetSplit, split_records
from malayalam_tts_dataset.dataset.writer import (
    write_dataset_csv,
    write_metadata_csv,
    write_run_summary,
)
from malayalam_tts_dataset.logging_utils import get_logger
from malayalam_tts_dataset.youtube.downloader import (
    YouTubeAudioDownloader,
    YouTubeDownloadSettings,
)


logger = get_logger(__name__)


@dataclass(frozen=True)
class PipelineResult:
    success: bool
    summary: dict[str, Any]


class DatasetBuildPipeline:
    def __init__(self, config: AppConfig, overwrite: bool | None = None) -> None:
        self.config = config
        self.overwrite = (
            config.runtime.overwrite_outputs
            if overwrite is None
            else bool(overwrite)
        )

        self.downloader = YouTubeAudioDownloader(
            YouTubeDownloadSettings(
                audio_full_dir=config.audio_full_dir,
                sample_rate=config.audio.sample_rate,
                channels=config.audio.channels,
                cookie_file=config.youtube.cookie_file,
                use_browser_cookies=config.youtube.use_browser_cookies,
                browser_for_cookies=config.youtube.browser_for_cookies,
                sleep_interval=config.youtube.ydl_sleep_interval,
                max_sleep_interval=config.youtube.ydl_max_sleep_interval,
                retries=config.youtube.retries,
                ratelimit=config.youtube.ratelimit,
                reuse_existing_audio=config.runtime.reuse_existing_audio,
                fetch_title_for_existing_audio=config.youtube.fetch_title_for_existing_audio,
                quiet=True,
            )
        )

        self.transcriber = WhisperTranscriber(
            model_name=config.asr.whisper_model,
            language=config.asr.language,
            keep_empty_segments=config.asr.keep_empty_segments,
        )

    @classmethod
    def from_config_path(
        cls,
        config_path: str | Path,
        overwrite: bool | None = None,
    ) -> "DatasetBuildPipeline":
        config = load_config(config_path)
        return cls(config=config, overwrite=overwrite)

    def run(self) -> PipelineResult:
        logger.info("Starting dataset build pipeline")
        logger.info("Output directory: %s", self.config.output_dir)

        video_results: list[dict[str, Any]] = []
        all_dataset_records: list[DatasetRecord] = []
        errors: list[dict[str, Any]] = []

        for index, url in enumerate(self.config.video_urls, start=1):
            logger.info("=" * 90)
            logger.info("Processing URL %s/%s: %s", index, len(self.config.video_urls), url)

            result = self._process_one_url(url=url, index=index)
            video_results.append(result)

            if result["success"]:
                all_dataset_records.extend(result["dataset_records"])
            else:
                errors.append(
                    {
                        "source_url": url,
                        "video_id": result.get("video_id"),
                        "stage": result.get("failed_stage"),
                        "error": result.get("error"),
                    }
                )

            self._sleep_between_items(index)

        logger.info("Writing dataset outputs")

        dataset_csv_path = write_dataset_csv(
            all_dataset_records,
            self.config.dataset_csv_path,
        )
        metadata_csv_path = write_metadata_csv(
            all_dataset_records,
            self.config.metadata_csv_path,
        )

        split = split_records(
            records=all_dataset_records,
            train_ratio=self.config.split.train_ratio,
            val_ratio=self.config.split.val_ratio,
            test_ratio=self.config.split.test_ratio,
            seed=self.config.split.random_seed,
        )

        split_paths = self._write_split_outputs(split)

        successful_videos = sum(1 for result in video_results if result["success"])
        failed_videos = len(video_results) - successful_videos
        total_duration = sum(record.duration_sec for record in all_dataset_records)
        average_duration = (
            total_duration / len(all_dataset_records)
            if all_dataset_records
            else 0.0
        )

        summary = {
            "total_urls": len(self.config.video_urls),
            "successful_videos": successful_videos,
            "failed_videos": failed_videos,
            "total_utterances": len(all_dataset_records),
            "total_duration_seconds": round(total_duration, 3),
            "average_utterance_duration": round(average_duration, 3),
            "split_strategy": split.strategy,
            "split_warnings": split.warnings,
            "splits": {
                "train_count": len(split.train),
                "val_count": len(split.val),
                "test_count": len(split.test),
            },
            "output_paths": {
                "dataset_csv": str(dataset_csv_path),
                "metadata_csv": str(metadata_csv_path),
                **{key: str(value) for key, value in split_paths.items()},
                "run_summary": str(self.config.run_summary_path),
            },
            "errors": errors,
            "video_results": self._strip_records_from_video_results(video_results),
        }

        run_summary_path = write_run_summary(summary, self.config.run_summary_path)
        summary["output_paths"]["run_summary"] = str(run_summary_path)

        self._log_summary(summary)

        return PipelineResult(
            success=failed_videos == 0,
            summary=summary,
        )

    def _process_one_url(self, url: str, index: int) -> dict[str, Any]:
        item_result: dict[str, Any] = {
            "source_url": url,
            "video_id": None,
            "download": None,
            "full_audio_validation": None,
            "trim": None,
            "trimmed_audio_validation": None,
            "transcription": None,
            "segmentation": None,
            "dataset_records": [],
            "success": False,
            "failed_stage": None,
            "error": None,
        }

        download_result = self.downloader.download(url)
        item_result["video_id"] = download_result.video_id or None
        item_result["download"] = {
            "video_id": download_result.video_id,
            "source_url": download_result.source_url,
            "wav_path": str(download_result.wav_path),
            "title": download_result.title,
            "success": download_result.success,
            "error": download_result.error,
            "used_existing": download_result.used_existing,
            "fresh_download": download_result.fresh_download,
            "download_attempted": download_result.download_attempted,
        }

        if download_result.used_existing:
            logger.info("Using existing WAV: %s", download_result.wav_path)
        elif download_result.fresh_download:
            logger.info("Fresh download successful: %s", download_result.wav_path)
        elif download_result.download_attempted and not download_result.success:
            logger.info("Download attempted and failed")

        if not download_result.success:
            return self._fail(item_result, "download", download_result.error)

        required_full_audio_duration = (
            self.config.audio.trim_start_seconds
            + self.config.audio.trim_duration_seconds
        )

        full_validation = validate_audio_file(
            download_result.wav_path,
            min_duration_sec=required_full_audio_duration,
        )
        item_result["full_audio_validation"] = validation_result_to_dict(full_validation)

        if not full_validation.is_valid:
            return self._fail(
                item_result,
                "full_audio_validation",
                "; ".join(full_validation.errors),
            )

        logger.info("Full audio validation successful")

        trimmed_wav_path = (
            self.config.audio_trimmed_dir / f"{download_result.video_id}_trimmed.wav"
        )

        trim_result = trim_audio_window(
            input_wav_path=download_result.wav_path,
            output_wav_path=trimmed_wav_path,
            start_seconds=self.config.audio.trim_start_seconds,
            duration_seconds=self.config.audio.trim_duration_seconds,
            overwrite=self.overwrite,
        )
        item_result["trim"] = trim_result_to_dict(trim_result)

        if not trim_result.success:
            return self._fail(item_result, "trim", trim_result.error)

        logger.info("Trim successful: %s", trim_result.output_path)

        trimmed_validation = validate_audio_file(
            trim_result.output_path,
            min_duration_sec=max(0.1, self.config.audio.trim_duration_seconds - 0.25),
        )
        item_result["trimmed_audio_validation"] = validation_result_to_dict(
            trimmed_validation
        )

        if not trimmed_validation.is_valid:
            return self._fail(
                item_result,
                "trimmed_audio_validation",
                "; ".join(trimmed_validation.errors),
            )

        logger.info("Trimmed audio validation successful")

        try:
            transcript_segments = self.transcriber.transcribe(trim_result.output_path)

            transcript_json_path = (
                self.config.transcripts_dir / f"{download_result.video_id}_segments.json"
            )

            save_transcript_json(
                video_id=download_result.video_id,
                source_path=trim_result.output_path,
                model_name=self.config.asr.whisper_model,
                language=self.config.asr.language,
                segments=transcript_segments,
                output_path=transcript_json_path,
            )

            item_result["transcription"] = {
                "success": True,
                "json_path": str(transcript_json_path),
                "model_name": self.config.asr.whisper_model,
                "language": self.config.asr.language,
                "segment_count": len(transcript_segments),
                "preview": [segment.text for segment in transcript_segments[:5]],
                "segments": transcript_segments_to_dicts(transcript_segments),
                "error": None,
            }

            logger.info("Transcription successful: %s segments", len(transcript_segments))

        except Exception as exc:
            logger.exception("Transcription failed")
            return self._fail(item_result, "transcription", str(exc))

        try:
            dataset_records = segment_transcribed_audio(
                trimmed_wav_path=trim_result.output_path,
                transcript_segments=transcript_segments,
                video_id=download_result.video_id,
                source_url=download_result.source_url,
                output_segments_dir=self.config.segments_dir,
                min_utterance_seconds=self.config.audio.min_utterance_seconds,
                max_utterance_seconds=self.config.audio.max_utterance_seconds,
                silence_thresh_dbfs=self.config.audio.silence_thresh_dbfs,
                min_silence_len_ms=self.config.audio.min_silence_len_ms,
                keep_silence_ms=self.config.audio.keep_silence_ms,
                overwrite=self.overwrite,
            )

            item_result["segmentation"] = {
                "success": True,
                "utterance_count": len(dataset_records),
                "records": dataset_records_to_dicts(dataset_records),
                "error": None,
            }
            item_result["dataset_records"] = dataset_records

            logger.info("Segmentation successful: %s utterances", len(dataset_records))

        except Exception as exc:
            logger.exception("Segmentation failed")
            return self._fail(item_result, "segmentation", str(exc))

        item_result["success"] = True
        return item_result

    def _write_split_outputs(self, split: DatasetSplit) -> dict[str, Path]:
        paths = {
            "train_csv": write_dataset_csv(split.train, self.config.train_csv_path),
            "val_csv": write_dataset_csv(split.val, self.config.val_csv_path),
            "test_csv": write_dataset_csv(split.test, self.config.test_csv_path),
            "train_metadata_csv": write_metadata_csv(
                split.train,
                self.config.train_metadata_csv_path,
            ),
            "val_metadata_csv": write_metadata_csv(
                split.val,
                self.config.val_metadata_csv_path,
            ),
            "test_metadata_csv": write_metadata_csv(
                split.test,
                self.config.test_metadata_csv_path,
            ),
        }
        return paths

    @staticmethod
    def _fail(
        item_result: dict[str, Any],
        stage: str,
        error: str | None,
    ) -> dict[str, Any]:
        item_result["success"] = False
        item_result["failed_stage"] = stage
        item_result["error"] = error or f"Failed at stage: {stage}"
        logger.error("Failed at stage '%s': %s", stage, item_result["error"])
        return item_result

    def _sleep_between_items(self, current_index: int) -> None:
        total_items = len(self.config.video_urls)
        if current_index >= total_items:
            return

        sleep_seconds = self.config.youtube.sleep_between_downloads
        if sleep_seconds <= 0:
            return

        logger.info("Sleeping %s seconds before next item", sleep_seconds)
        time.sleep(sleep_seconds)

    @staticmethod
    def _strip_records_from_video_results(
        video_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        clean_results = []

        for result in video_results:
            clean_result = dict(result)
            records = clean_result.pop("dataset_records", [])
            clean_result["dataset_record_count"] = len(records)
            clean_results.append(clean_result)

        return clean_results

    @staticmethod
    def _log_summary(summary: dict[str, Any]) -> None:
        logger.info("=" * 90)
        logger.info("PIPELINE SUMMARY")
        logger.info("Total URLs                 : %s", summary["total_urls"])
        logger.info("Successful videos          : %s", summary["successful_videos"])
        logger.info("Failed videos              : %s", summary["failed_videos"])
        logger.info("Total utterances           : %s", summary["total_utterances"])
        logger.info("Total duration seconds     : %s", summary["total_duration_seconds"])
        logger.info("Average utterance duration : %s", summary["average_utterance_duration"])
        logger.info("Split strategy             : %s", summary["split_strategy"])
        logger.info("Train count                : %s", summary["splits"]["train_count"])
        logger.info("Val count                  : %s", summary["splits"]["val_count"])
        logger.info("Test count                 : %s", summary["splits"]["test_count"])

        for warning in summary["split_warnings"]:
            logger.warning("Split warning: %s", warning)

        logger.info("Run summary                : %s", summary["output_paths"]["run_summary"])