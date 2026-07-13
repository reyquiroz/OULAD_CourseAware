"""
check_data.py — Preflight data file check for the OULAD pipeline.

Verifies that all required raw CSV files are present in DATA_DIR before
running the pipeline.  Prints a ✓/✗ status line for each file and raises
FileNotFoundError with the OULAD download URL if any required file is missing.

Usage
-----
    python src/check_data.py            # check default DATA_DIR (data/raw/)
    python src/check_data.py --data-dir /path/to/data/raw

Called automatically at the top of run_evaluation.py and run_graph_pipeline.py.
"""

import argparse
import sys
from pathlib import Path

# Ensure src/ is importable when run as a top-level script
sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_DIR

# Required files with approximate sizes for user guidance.
# studentVle.csv is gitignored (433 MB) and must be downloaded separately.
REQUIRED_FILES = [
    ("studentInfo.csv",         "~3 MB",   False),
    ("studentVle.csv",          "~433 MB", True),   # gitignored — must download
    ("studentAssessment.csv",   "~6 MB",   False),
    ("assessments.csv",         "~10 KB",  False),
    ("courses.csv",             "~1 KB",   False),
    ("vle.csv",                 "~500 KB", False),
    ("studentRegistration.csv", "~1.5 MB", False),
]

DOWNLOAD_URL = "https://analyse.kmi.open.ac.uk/open_dataset"


def check_data_files(data_dir=None) -> bool:
    """Check that all required OULAD CSV files exist in *data_dir*.

    Prints a ✓/✗ status line for each required file.  Returns True if all
    files are present, False otherwise.

    Args:
        data_dir: Path to the raw data directory.  Defaults to DATA_DIR
                  from config.py (``data/raw/`` relative to project root).

    Raises:
        FileNotFoundError: if any required file is missing, with the OULAD
                           download URL included in the message.
    """
    if data_dir is None:
        data_dir = DATA_DIR
    data_dir = Path(data_dir)

    missing = []
    print(f"Checking data files in: {data_dir}")
    print()

    for filename, size_hint, gitignored in REQUIRED_FILES:
        path = data_dir / filename
        exists = path.exists()
        note = " (gitignored — download separately)" if gitignored else ""
        status = "✓" if exists else "✗"
        print(f"  {status} {filename:<30} {size_hint:<10}{note}")
        if not exists:
            missing.append(filename)

    print()

    if missing:
        msg = (
            f"\nMissing {len(missing)} required file(s) in {data_dir}:\n"
            + "\n".join(f"  - {f}" for f in missing)
            + f"\n\nDownload the OULAD dataset from:\n  {DOWNLOAD_URL}\n"
            + "Place all CSV files in the data/raw/ directory before running "
            "the pipeline."
        )
        raise FileNotFoundError(msg)

    print("All required data files present. ✓")
    return True


def main():
    p = argparse.ArgumentParser(
        description="Check that all required OULAD CSV data files are present."
    )
    p.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to raw data directory (default: data/raw/ from config).",
    )
    args = p.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else None
    try:
        check_data_files(data_dir)
        sys.exit(0)
    except FileNotFoundError as e:
        print(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
