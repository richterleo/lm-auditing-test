# logging_config.py
import logging

from rich.console import Console
from rich.logging import RichHandler

from logging.config import dictConfig
from typing import Optional
from pathlib import Path

# console = Console()
# init_context = console.status("[bold green]Initializing the SFTTrainer...")


def setup_logging(
    *args,
    default_level=logging.INFO,
    log_file: Optional[str] = None,
    log_format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    tag="test_results",
    directory="logs",
    use_rich=True,  # Added to toggle rich logging
):
    """Sets up logging configuration with optional rich logging."""

    # Create logs directory if it doesn't exist
    if not Path(directory).exists():
        Path(directory).mkdir(parents=True, exist_ok=True)

    # Determine the log file name
    if log_file:
        log_file = Path(directory) / log_file
    else:
        log_file = tag
        for item in args:
            if item:
                log_file = log_file + "_" + str(item)
        log_file = Path(directory) / f"{log_file}.log"

    # Choose the console handler
    if use_rich:
        console_handler = {
            "class": "rich.logging.RichHandler",
            "formatter": "default",
            "rich_tracebacks": True,  # Enable rich tracebacks for better error reporting
            "markup": True,  # Enable markup in logging messages
        }
    else:
        console_handler = {
            "class": "logging.StreamHandler",
            "formatter": "default",
        }

    # Logging configuration dictionary
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": log_format,
            },
        },
        "handlers": {
            "console": console_handler,
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

    # Apply the logging configuration
    dictConfig(logging_config)
