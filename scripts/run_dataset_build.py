from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _bootstrap_src_path() -> None:
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    src_path = project_root / "src"

    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


_bootstrap_src_path()

from malayalam_tts_dataset.config import ConfigError  # noqa: E402
from malayalam_tts_dataset.logging_utils import get_logger, setup_logging  # noqa: E402
from malayalam_tts_dataset.pipeline.build_dataset import DatasetBuildPipeline  # noqa: E402


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Malayalam TTS dataset from YouTube video URLs."
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Path to YAML config. "
            "Defaults to CONFIG_PATH env var or configs/default.yaml."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite generated outputs for trimming and segmentation.",
    )

    return parser.parse_args()


def resolve_config_path(cli_config_path: str | None) -> Path:
    if cli_config_path:
        return Path(cli_config_path)

    env_config_path = os.getenv("CONFIG_PATH")
    if env_config_path:
        return Path(env_config_path)

    return Path("configs/default.yaml")


def main() -> int:
    args = parse_args()

    setup_logging()

    config_path = resolve_config_path(args.config)

    logger.info("Starting Malayalam TTS Dataset Builder")
    logger.info("Config path: %s", config_path)
    logger.info("Overwrite: %s", args.overwrite)

    try:
        pipeline = DatasetBuildPipeline.from_config_path(
            config_path=config_path,
            overwrite=args.overwrite if args.overwrite else None,
        )

        result = pipeline.run()

    except ConfigError as exc:
        logger.error("Invalid configuration: %s", exc)
        return 2
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 130
    except Exception as exc:
        logger.exception("Unexpected pipeline failure: %s", exc)
        return 1

    print("\nRun summary:")
    print(json.dumps(result.summary, indent=2, ensure_ascii=False))

    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())