# logging_config.py
import logging

from rich.console import Console
from rich.logging import RichHandler

from logging.config import dictConfig
from typing import Optional, Any
from pathlib import Path

# console = Console()
# init_context = console.status("[bold green]Initializing the SFTTrainer...")


def setup_logging(
    *args: Any,
    default_level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    tag: str = "test_results",
    directory: str = "logs",
    use_rich: bool = True,
    quiet: bool = True,
) -> Optional[Path]:
    """Sets up logging configuration with optional rich logging.

    Returns:
        Optional[Path]: The path to the log file if one is created, otherwise None.
    """

    handlers = {}
    root_handlers = []
    log_file_path = None

    if quiet:
        # Create logs directory if it doesn't exist
        log_dir = Path(directory)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Determine the log file name
        if log_file:
            log_file_path = log_dir / log_file
        else:
            log_file_name = tag
            for item in args:
                if item:
                    log_file_name = log_file_name + "_" + str(item)
            log_file_path = log_dir / f"{log_file_name}.log"

        handlers["file"] = {
            "class": "logging.FileHandler",
            "formatter": "default",
            "filename": log_file_path,
        }
        root_handlers.append("file")

    else:  # Log to console only
        console_handler_config = {
            "class": "rich.logging.RichHandler" if use_rich else "logging.StreamHandler",
            "formatter": "default",
        }
        if use_rich:
            console_handler_config.update(
                {
                    "rich_tracebacks": True,
                    "markup": True,
                }
            )
        handlers["console"] = console_handler_config
        root_handlers.append("console")

    # Logging configuration dictionary
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": log_format,
            },
        },
        "handlers": handlers,
        "root": {
            "handlers": root_handlers,
            "level": default_level,
        },
        "loggers": {
            "": {  # root logger
                "level": default_level,
                "handlers": root_handlers,
            },
            "__main__": {
                "level": default_level,
                "handlers": root_handlers,
                "propagate": False,
            },
        },
    }

    # Apply the logging configuration
    dictConfig(logging_config)
    return log_file_path
