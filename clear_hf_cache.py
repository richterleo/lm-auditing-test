#!/usr/bin/env python3
"""
Utility script to manage HuggingFace cache and free up disk space.
Use this if you're running out of disk space.
"""

import os
import shutil
from pathlib import Path
from huggingface_hub import scan_cache_dir


def get_cache_info():
    """Get information about HuggingFace cache usage."""
    try:
        cache_info = scan_cache_dir()
        print(f"Cache directory: {cache_info.cache_dir}")
        print(f"Cache size: {cache_info.size_on_disk_str}")
        print(f"Number of repos: {len(cache_info.repos)}")
        return cache_info
    except Exception as e:
        print(f"Error scanning cache: {e}")
        return None


def clean_cache():
    """Clean up HuggingFace cache to free disk space."""
    cache_info = get_cache_info()
    if not cache_info:
        return

    print("\nCleaning cache...")
    try:
        # Clean up cache
        delete_strategy = cache_info.delete_revisions(*[repo.revisions for repo in cache_info.repos])
        print(f"Will free {delete_strategy.expected_freed_size_str}")

        # Actually delete
        delete_strategy.execute()
        print("Cache cleaned successfully!")

        # Show new cache info
        print("\nAfter cleanup:")
        get_cache_info()

    except Exception as e:
        print(f"Error cleaning cache: {e}")


def main():
    print("HuggingFace Cache Manager")
    print("=" * 30)

    # Show current cache info
    get_cache_info()

    # Ask user if they want to clean
    response = input("\nDo you want to clean the cache? (y/N): ")
    if response.lower() in ["y", "yes"]:
        clean_cache()
    else:
        print("Cache not cleaned.")


if __name__ == "__main__":
    main()
