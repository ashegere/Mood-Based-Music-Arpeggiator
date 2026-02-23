#!/usr/bin/env python3
"""
Sync training files from arpeggio_mood_data_updated.json.

Iterates through the JSON manifest and ensures all referenced MIDI files
exist in the training folder. If a file is missing, searches the computer
and moves it to the training folder.

Usage:
    python scripts/sync_training_files.py

    # Dry run (show what would be done)
    python scripts/sync_training_files.py --dry-run

    # Custom paths
    python scripts/sync_training_files.py --json-path data/training/arpeggio_mood_data_updated.json \
        --training-dir data/training/midis --search-root /Users/abel
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_JSON_PATH = "data/training/arpeggio_mood_data_updated.json"
DEFAULT_TRAINING_DIR = "data/training/augmented"
DEFAULT_SEARCH_ROOT = str(Path.home())

# Directories to skip during search (for performance)
SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    "Library", "Applications", ".Trash", "Pictures", "Movies",
}


# =============================================================================
# File Search Functions
# =============================================================================

def find_file_mdfind(filename: str, search_root: str = "/") -> Optional[str]:
    """
    Find a file using macOS Spotlight (mdfind).

    Fast but only works on macOS and indexed locations.

    Args:
        filename: Name of the file to find.
        search_root: Root directory to search in.

    Returns:
        Full path to the file, or None if not found.
    """
    try:
        result = subprocess.run(
            ["mdfind", "-name", filename],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0 and result.stdout.strip():
            paths = result.stdout.strip().split("\n")
            # Filter to search_root if specified
            for path in paths:
                if path.startswith(search_root):
                    return path
            # Return first result if no match in search_root
            return paths[0] if paths else None

    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return None


def find_file_walk(
    filename: str,
    search_root: str,
    skip_dirs: set = SKIP_DIRS,
) -> Optional[str]:
    """
    Find a file using os.walk (slower but portable).

    Args:
        filename: Name of the file to find.
        search_root: Root directory to search in.
        skip_dirs: Directory names to skip.

    Returns:
        Full path to the file, or None if not found.
    """
    search_root = Path(search_root)

    for root, dirs, files in os.walk(search_root):
        # Skip specified directories
        dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith(".")]

        if filename in files:
            return str(Path(root) / filename)

    return None


def find_file(
    filename: str,
    search_root: str,
    use_spotlight: bool = True,
) -> Optional[str]:
    """
    Find a file on the computer.

    Tries Spotlight first (fast), falls back to os.walk.

    Args:
        filename: Name of the file to find.
        search_root: Root directory to search in.
        use_spotlight: Whether to try Spotlight first.

    Returns:
        Full path to the file, or None if not found.
    """
    # Try Spotlight first on macOS
    if use_spotlight and sys.platform == "darwin":
        result = find_file_mdfind(filename, search_root)
        if result:
            return result

    # Fall back to os.walk
    return find_file_walk(filename, search_root)


# =============================================================================
# Sync Logic
# =============================================================================

def load_manifest(json_path: str) -> List[dict]:
    """Load the JSON manifest file."""
    with open(json_path, "r") as f:
        return json.load(f)


def sync_files(
    json_path: str,
    training_dir: str,
    search_root: str,
    dry_run: bool = False,
    verbose: bool = True,
) -> Tuple[int, int, int]:
    """
    Sync files from manifest to training directory.

    Args:
        json_path: Path to the JSON manifest.
        training_dir: Target directory for training files.
        search_root: Root directory to search for missing files.
        dry_run: If True, don't actually move files.
        verbose: If True, print progress.

    Returns:
        Tuple of (found_count, missing_count, moved_count)
    """
    # Load manifest
    manifest = load_manifest(json_path)
    if verbose:
        print(f"Loaded {len(manifest)} entries from manifest")

    # Ensure training directory exists
    training_path = Path(training_dir)
    if not dry_run:
        training_path.mkdir(parents=True, exist_ok=True)

    # Track statistics
    found_count = 0
    missing_count = 0
    moved_count = 0
    not_found_files = []

    # Get unique filenames
    filenames = list(set(entry["Filename"] for entry in manifest))
    if verbose:
        print(f"Checking {len(filenames)} unique files...")

    for i, filename in enumerate(filenames):
        target_path = training_path / filename

        # Check if file already exists in training dir
        if target_path.exists():
            found_count += 1
            continue

        # File is missing, try to find it
        if verbose:
            print(f"[{i+1}/{len(filenames)}] Missing: {filename}", end=" ")

        source_path = find_file(filename, search_root)

        if source_path:
            if verbose:
                print(f"-> Found: {source_path}")

            if not dry_run:
                # Copy file to training directory
                shutil.copy2(source_path, target_path)

            moved_count += 1
        else:
            if verbose:
                print("-> NOT FOUND")
            missing_count += 1
            not_found_files.append(filename)

    # Summary
    if verbose:
        print("\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)
        print(f"Already in training dir: {found_count}")
        print(f"Found and {'would move' if dry_run else 'moved'}: {moved_count}")
        print(f"Not found on computer: {missing_count}")

        if not_found_files:
            print("\nFiles not found:")
            for f in not_found_files[:20]:
                print(f"  - {f}")
            if len(not_found_files) > 20:
                print(f"  ... and {len(not_found_files) - 20} more")

    return found_count, missing_count, moved_count


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Sync training files from JSON manifest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--json-path", "-j",
        type=str,
        default=DEFAULT_JSON_PATH,
        help="Path to arpeggio_mood_data_updated.json",
    )

    parser.add_argument(
        "--training-dir", "-t",
        type=str,
        default=DEFAULT_TRAINING_DIR,
        help="Target directory for training files",
    )

    parser.add_argument(
        "--search-root", "-s",
        type=str,
        default=DEFAULT_SEARCH_ROOT,
        help="Root directory to search for missing files",
    )

    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be done without making changes",
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output",
    )

    args = parser.parse_args()

    # Validate JSON path exists
    if not Path(args.json_path).exists():
        print(f"Error: JSON file not found: {args.json_path}")
        sys.exit(1)

    if args.dry_run:
        print("DRY RUN - No files will be moved\n")

    # Run sync
    sync_files(
        json_path=args.json_path,
        training_dir=args.training_dir,
        search_root=args.search_root,
        dry_run=args.dry_run,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
