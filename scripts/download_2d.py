"""Download required data for blocked_stacking scenarios.

This script downloads five separate zip files from Google Drive:
- logs.zip: Training logs
- pred_nets.zip: Predicate networks
- trained_policies.zip: Trained skill policies
- training_data.zip: Planner datasets
- videos.zip: Demo videos

Usage:
    python scripts/download_2d.py
    python scripts/download_2d.py --skip-logs --skip-videos
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Optional

try:
    import gdown  # type: ignore[import-untyped]
except ImportError:
    print("Error: gdown library not found. Install with: pip install gdown")
    sys.exit(1)


def extract_file_id(url_or_id: str) -> str:
    """Extract Google Drive file ID from URL or return ID if already extracted.

    Args:
        url_or_id: Either a Google Drive URL or a file ID

    Returns:
        The file ID string

    Examples:
        >>> extract_file_id("https://drive.google.com/file/d/xx/view?usp=sharing")
        '1lWfzy6f'
        >>> extract_file_id("1lWfzy6f")
        '1lWfzy6f'
    """
    # If it's already just an ID (no slashes or domains), return it
    if "/" not in url_or_id and "drive.google.com" not in url_or_id:
        return url_or_id

    # Extract from /d/FILE_ID/ or /d/FILE_ID? patterns
    match = re.search(r"/d/([a-zA-Z0-9_-]+)", url_or_id)
    if match:
        return match.group(1)

    # Extract from id=FILE_ID pattern
    match = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url_or_id)
    if match:
        return match.group(1)

    raise ValueError(f"Could not extract file ID from: {url_or_id}")


def download_file_from_google_drive(
    file_id: str, destination: Path, verbose: bool = True
) -> None:
    """Download a file from Google Drive using gdown.

    Args:
        file_id: Google Drive file ID
        destination: Path where the file should be saved
        verbose: Whether to print progress messages
    """
    if verbose:
        print(f"Downloading from Google Drive (ID: {file_id})")
        print(f"  -> {destination}")

    # Create parent directory if it doesn't exist
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Build Google Drive URL
    url = f"https://drive.google.com/uc?id={file_id}"

    try:
        # Use gdown to download the file (handles large files automatically)
        gdown.download(url, str(destination), quiet=not verbose, fuzzy=True)

        if verbose and destination.exists():
            size_mb = destination.stat().st_size / (1024 * 1024)
            print(f"  Downloaded: {size_mb:.2f} MB")

    except Exception as e:
        raise RuntimeError(f"Failed to download file {file_id}: {e}") from e


def extract_zip(
    zip_path: Path,
    extract_to: Path,
    target_dirs: list[str],
    verbose: bool = True,
) -> None:
    """Extract a zip file and ensure target directories are at the root level.

    If the zip contains a single parent folder with the target directories inside,
    this function will move the contents up to the extract_to level.

    Args:
        zip_path: Path to the zip file
        extract_to: Directory to extract files to
        target_dirs: List of expected directory names
        verbose: Whether to print progress messages
    """
    if verbose:
        print(f"Extracting {zip_path.name}")
        print(f"  -> {extract_to}")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        # Extract to a temporary location first
        temp_extract = extract_to / ".temp_extract"
        temp_extract.mkdir(exist_ok=True)

        zip_ref.extractall(temp_extract)
        file_count = len(zip_ref.namelist())

        if verbose:
            print(f"  Extracted {file_count} files")

    # Check if we have a nested structure (single parent folder)
    extracted_items = list(temp_extract.iterdir())

    # If there's only one directory and it contains our target directories
    if len(extracted_items) == 1 and extracted_items[0].is_dir():
        parent_folder = extracted_items[0]
        # Check if target directories are inside this parent folder
        has_targets_inside = any(
            (parent_folder / target_dir).exists() for target_dir in target_dirs
        )

        if has_targets_inside:
            if verbose:
                print(
                    f"  Found nested structure in '{parent_folder.name}', flattening..."
                )

            # Move contents from parent folder to extract_to
            for item in parent_folder.iterdir():
                dest = extract_to / item.name
                if dest.exists():
                    if dest.is_dir():
                        shutil.rmtree(dest)
                    else:
                        dest.unlink()
                shutil.move(str(item), str(extract_to))

            # Remove the now-empty parent folder and temp directory
            shutil.rmtree(temp_extract)
            if verbose:
                print(f"  Moved contents to root level")
        else:
            # No nesting, just move everything to extract_to
            for item in temp_extract.iterdir():
                dest = extract_to / item.name
                if dest.exists():
                    if dest.is_dir():
                        shutil.rmtree(dest)
                    else:
                        dest.unlink()
                shutil.move(str(item), str(extract_to))
            shutil.rmtree(temp_extract)
    else:
        # Multiple items at root or files - move everything
        for item in temp_extract.iterdir():
            dest = extract_to / item.name
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            shutil.move(str(item), str(extract_to))

        # Clean up temp directory
        shutil.rmtree(temp_extract)


def download_2d_data(
    file_ids: Optional[dict[str, str]] = None,
    repo_root: Optional[Path] = None,
    skip_files: Optional[list[str]] = None,
    verbose: bool = True,
) -> None:
    """Download all required data for blocked_stacking scenarios.

    This function downloads five separate zip files from Google Drive:
    - logs.zip -> logs/
    - pred_nets.zip -> pred_nets/
    - trained_policies.zip -> trained_policies/
    - training_data.zip -> training_data/
    - videos.zip -> videos/

    Args:
        file_ids: Dict mapping zip filenames to Google Drive file IDs/URLs.
                  If None, uses default file IDs.
        repo_root: Root directory of the repository.
        skip_files: List of filenames to skip (e.g., ['logs.zip', 'videos.zip'])
        verbose: Whether to print progress messages

    Raises:
        FileNotFoundError: If repo_root cannot be determined
        Exception: If download or extraction fails
    """
    # Determine repository root
    if repo_root is None:
        # Assume script is in scripts/ directory
        script_dir = Path(__file__).parent
        repo_root = script_dir.parent

    if not repo_root.exists():
        raise FileNotFoundError(f"Repository root not found: {repo_root}")

    if verbose:
        print(f"Repository root: {repo_root}")
        print()

    # Default Google Drive file IDs for each zip file
    if file_ids is None:
        file_ids = {
            "logs.zip": "PLACEHOLDER_LOGS_FILE_ID",
            "pred_nets.zip": "PLACEHOLDER_PRED_NETS_FILE_ID",
            "trained_policies.zip": "PLACEHOLDER_TRAINED_POLICIES_FILE_ID",
            "training_data.zip": "PLACEHOLDER_TRAINING_DATA_FILE_ID",
            "videos.zip": "PLACEHOLDER_VIDEOS_FILE_ID",
        }

    # Define expected directories after extraction (without .zip extension)
    expected_dirs = {
        "logs.zip": ["logs"],
        "pred_nets.zip": ["pred_nets"],
        "trained_policies.zip": ["trained_policies"],
        "training_data.zip": ["training_data"],
        "videos.zip": ["videos"],
    }

    if skip_files is None:
        skip_files = []

    files_to_download = [f for f in file_ids.keys() if f not in skip_files]

    if verbose:
        print(f"{'=' * 60}")
        print(f"Downloading {len(files_to_download)} zip files")
        print(f"{'=' * 60}")
        print()

    downloaded_dirs = []

    for zip_filename in files_to_download:
        file_id = file_ids[zip_filename]
        zip_path = repo_root / zip_filename
        target_dirs = expected_dirs[zip_filename]

        # Check if target directory already exists
        existing_dirs = [d for d in target_dirs if (repo_root / d).exists()]
        if existing_dirs:
            if verbose:
                print(
                    f"[{zip_filename}] Skipping - directory already exists: {', '.join(existing_dirs)}"
                )
                print()
            continue

        try:
            if verbose:
                print(f"[{zip_filename}] Starting download...")

            # Extract file ID from URL if necessary
            file_id_extracted = extract_file_id(file_id)

            # Download from Google Drive
            download_file_from_google_drive(
                file_id_extracted, zip_path, verbose=verbose
            )

            # Extract with automatic flattening if nested
            extract_zip(zip_path, repo_root, target_dirs, verbose=verbose)

            # Clean up zip file
            zip_path.unlink()
            if verbose:
                print(f"  Removed {zip_path.name}")
                print()

            downloaded_dirs.extend(target_dirs)

        except Exception as e:
            print(f"Error downloading {zip_filename}: {e}")
            # Clean up partial zip file if it exists
            if zip_path.exists():
                zip_path.unlink()
            raise

    if verbose:
        print(f"{'=' * 60}")
        print("Download complete!")
        print(f"{'=' * 60}")
        print()
        if downloaded_dirs:
            print("Extracted directories:")
            for dir_name in downloaded_dirs:
                if (repo_root / dir_name).exists():
                    print(f"  - {dir_name}")
        print()
        print("You can now uncomment and run tests with:")
        print(
            "  pytest tests/approaches/test_lifelong_ref.py::"
            "test_loading_learned_skill_predicate_blocked_stacking_... -xvs"
        )


def main() -> int:
    """Main entry point for the download script."""
    parser = argparse.ArgumentParser(
        description="Download required data for blocked_stacking scenarios",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all files with default settings
  python scripts/download_2d.py

  # Skip certain files (e.g., large video files)
  python scripts/download_2d.py --skip-logs --skip-videos

  # Provide custom Google Drive file IDs
  python scripts/download_2d.py --logs-id <file_id> --pred-nets-id <file_id>

  # Quiet mode
  python scripts/download_2d.py --quiet

This script downloads five zip files from Google Drive:
  - logs.zip: Training logs
  - pred_nets.zip: Predicate networks
  - trained_policies.zip: Trained skill policies
  - training_data.zip: Planner datasets
  - videos.zip: Demo videos

Each file is extracted and the zip is removed automatically.
        """,
    )

    parser.add_argument(
        "--logs-id",
        type=str,
        default="1EFAoLiXwkie7vva_fo32OFp95XOaCWjX",
        help="Google Drive file ID or URL for logs.zip",
    )

    parser.add_argument(
        "--pred-nets-id",
        type=str,
        default="1KFihu3l8ocRt36gh3uGP_2ZMbHHEA0CG",
        help="Google Drive file ID or URL for pred_nets.zip",
    )

    parser.add_argument(
        "--trained-policies-id",
        type=str,
        default="1pjkAwG3EBnY59AnuMFTZG1suTYm05p1_",
        help="Google Drive file ID or URL for trained_policies.zip",
    )

    parser.add_argument(
        "--training-data-id",
        type=str,
        default="1E2Hdw2nUiNVwmJzbT-U3emnQ9Abn3je2",
        help="Google Drive file ID or URL for training_data.zip",
    )

    parser.add_argument(
        "--videos-id",
        type=str,
        default="11fFwSSun8s8ZrYQkNDNBsYx1ze8qDKWU",
        help="Google Drive file ID or URL for videos.zip",
    )

    parser.add_argument(
        "--skip-logs",
        action="store_true",
        help="Skip downloading logs.zip",
    )

    parser.add_argument(
        "--skip-pred-nets",
        action="store_true",
        help="Skip downloading pred_nets.zip",
    )

    parser.add_argument(
        "--skip-trained-policies",
        action="store_true",
        help="Skip downloading trained_policies.zip",
    )

    parser.add_argument(
        "--skip-training-data",
        action="store_true",
        help="Skip downloading training_data.zip",
    )

    parser.add_argument(
        "--skip-videos",
        action="store_true",
        help="Skip downloading videos.zip",
    )

    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root directory (default: auto-detect from script location)",
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress messages",
    )

    args = parser.parse_args()

    # Build file_ids dict from provided arguments
    file_ids = {}
    if args.logs_id:
        file_ids["logs.zip"] = args.logs_id
    if args.pred_nets_id:
        file_ids["pred_nets.zip"] = args.pred_nets_id
    if args.trained_policies_id:
        file_ids["trained_policies.zip"] = args.trained_policies_id
    if args.training_data_id:
        file_ids["training_data.zip"] = args.training_data_id
    if args.videos_id:
        file_ids["videos.zip"] = args.videos_id

    # Build skip list from arguments
    skip_files = []
    if args.skip_logs:
        skip_files.append("logs.zip")
    if args.skip_pred_nets:
        skip_files.append("pred_nets.zip")
    if args.skip_trained_policies:
        skip_files.append("trained_policies.zip")
    if args.skip_training_data:
        skip_files.append("training_data.zip")
    if args.skip_videos:
        skip_files.append("videos.zip")

    try:
        download_2d_data(
            file_ids=file_ids if file_ids else None,
            repo_root=args.repo_root,
            skip_files=skip_files if skip_files else None,
            verbose=not args.quiet,
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
