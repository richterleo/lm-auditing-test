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
    quiet=True,  # Controls whether to show logs in console
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
            "file": {
                "class": "logging.FileHandler",
                "formatter": "default",
                "filename": log_file,
            },
        },
        "root": {
            "handlers": ["file"],
            "level": default_level,
        },
        "loggers": {
            "": {  # root logger
                "level": default_level,
                "handlers": ["file"],
            },
            "__main__": {
                "level": default_level,
                "handlers": ["file"],
                "propagate": False,
            },
        },
    }

    # Add console handler if not quiet
    if not quiet:
        console_handler = {
            "class": "rich.logging.RichHandler" if use_rich else "logging.StreamHandler",
            "formatter": "default",
            "rich_tracebacks": True,
            "markup": True,
        } if use_rich else {
            "class": "logging.StreamHandler",
            "formatter": "default",
        }
        
        logging_config["handlers"]["console"] = console_handler
        logging_config["root"]["handlers"].append("console")
        logging_config["loggers"][""]["handlers"].append("console")
        logging_config["loggers"]["__main__"]["handlers"].append("console")

    # Apply the logging configuration
    dictConfig(logging_config)
