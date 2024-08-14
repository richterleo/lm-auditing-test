# logging_config.py
import logging

from logging.config import dictConfig
from typing import Optional
from pathlib import Path


def setup_logging(
    model_name1,
    seed1,
    model_name2: Optional[str] = None,
    seed2: Optional[str] = None,
    fold_size: Optional[int] = None,
    epsilon: Optional[float] = None,
    metric: Optional[str] = None,
    default_level=logging.INFO,
    log_file: Optional[str] = None,
    log_format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    tag="test_results",
    directory="logs",
):
    directory = "logs"
    # TODO: replace by pathlib
    if not Path(directory).exists():
        Path(directory).mkdir(parents=True, exist_ok=True)

    if log_file:
        log_file = Path(directory) / log_file

    else:
        if (
            tag == "test_results"
            or tag == "test_results_and_analyze"
            or tag == "analyze"
        ):
            log_file = (
                Path(directory)
                / f"{tag}_{model_name1}_{seed1}_{model_name2}_{seed2}_{fold_size}_epsilon_{epsilon}.log"
            )

        elif tag == "evaluation":
            log_file = Path(directory) / f"{metric}_scores_{model_name1}_{seed1}.log"

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": log_format,
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
            },
            "file": {
                "class": "logging.FileHandler",
                "formatter": "default",
                "filename": log_file,
            },
        },
        "root": {
            "handlers": ["console", "file"],
            "level": default_level,
        },
        "loggers": {
            "": {  # root logger
                "level": default_level,
                "handlers": ["console", "file"],
            },
            "__main__": {
                "level": default_level,
                "handlers": ["console", "file"],
                "propagate": False,
            },
        },
    }

    dictConfig(logging_config)
