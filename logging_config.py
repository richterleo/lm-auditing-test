# logging_config.py
import logging

from logging.config import dictConfig
from typing import Optional
from pathlib import Path


def setup_logging(
    *args,
    default_level=logging.INFO,
    log_file: Optional[str] = None,
    log_format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    tag="test_results",
    directory="logs",
):
    """ """
    # TODO: replace by pathlib
    if not Path(directory).exists():
        Path(directory).mkdir(parents=True, exist_ok=True)

    if log_file:
        log_file = Path(directory) / log_file

    else:
        log_file = tag
        for item in args:
            if item:
                log_file = log_file + str(item)

        log_file = Path(directory) / f"{log_file}.log"

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
