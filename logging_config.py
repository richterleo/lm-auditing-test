# logging_config.py
import logging

from logging.config import dictConfig
from typing import Optional
from pathlib import Path


def setup_logging(
    model_name1,
    seed1,
    model_name2,
    seed2,
    fold_size,
    epsilon,
    default_level=logging.INFO,
    log_file: Optional[str] = None,
    log_format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    tag="test_results",
):
    directory = "logs"
    # TODO: replace by pathlib
    if not Path(directory).exists():
        Path(directory).mkdir(parents=True, exist_ok=True)

    log_file = (
        (
            Path(directory)
            / f"{tag}_{model_name1}_{seed1}_{model_name2}_{seed2}_{fold_size}_epsilon_{epsilon}.log"
        )
        if not log_file
        else log_file
    )

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
