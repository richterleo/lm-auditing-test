"""
Environment variable loader utility.
Loads environment variables from .env files if they exist.
"""

import os
from pathlib import Path
from typing import Optional


def load_env_file(env_file_path: Optional[Path] = None) -> None:
    """
    Load environment variables from a .env file.

    Args:
        env_file_path: Path to the .env file. If None, looks for .env in the project root.
    """
    if env_file_path is None:
        # Look for .env file in project root (3 levels up from this file)
        project_root = Path(__file__).resolve().parents[3]
        env_file_path = project_root / ".env"

    if not env_file_path.exists():
        return

    with open(env_file_path, "r") as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            # Parse key=value pairs
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()

                # Remove quotes if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]

                # Only set if not already in environment
                if key not in os.environ:
                    os.environ[key] = value


def get_required_env_var(var_name: str, description: str = "") -> str:
    """
    Get a required environment variable, raising an error if not found.

    Args:
        var_name: Name of the environment variable
        description: Description for error message

    Returns:
        The environment variable value

    Raises:
        ValueError: If the environment variable is not set
    """
    value = os.getenv(var_name)
    if not value:
        error_msg = f"Required environment variable '{var_name}' not set."
        if description:
            error_msg += f" {description}"
        error_msg += f"\nPlease copy .env.example to .env and fill in your API keys."
        raise ValueError(error_msg)
    return value


def validate_api_keys() -> bool:
    """
    Validate that all required API keys are present.

    Returns:
        True if all required keys are present, False otherwise
    """
    required_keys = {
        "WANDB_API_KEY": "Get from https://wandb.ai/settings",
        "HF_TOKEN": "Get from https://huggingface.co/settings/tokens",
        "PERSPECTIVE_API_KEY": "Get from https://developers.perspectiveapi.com/s/docs-get-started",
    }

    missing_keys = []
    for key, help_text in required_keys.items():
        if not os.getenv(key):
            missing_keys.append(f"  - {key}: {help_text}")

    if missing_keys:
        print("❌ Missing required API keys:")
        for key in missing_keys:
            print(key)
        print("\nPlease copy .env.example to .env and fill in your API keys.")
        return False

    print("✅ All required API keys are present")
    return True


# Auto-load .env file when this module is imported
load_env_file()
