"""Record machine and package versions next to the results.

Local runs and later server runs are compared against this file, so numeric
differences can be traced to library or hardware changes.

Usage:
    python src/record_environment.py
"""

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import RESULTS_DIR  # noqa: E402

ENVIRONMENT_FILE = RESULTS_DIR / "run_environment.json"
PACKAGES = [
    "torch",
    "torch_geometric",
    "scikit-learn",
    "pandas",
    "numpy",
    "pyarrow",
    "lightgbm",
    "openpyxl",
    "scipy",
]


def _sysctl(key: str) -> str:
    """Return a macOS sysctl value, or an empty string when unavailable."""
    try:
        return subprocess.run(
            ["sysctl", "-n", key], capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def collect_environment() -> dict:
    """Collect hardware, OS, Python and package versions."""
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None

    import torch

    mem_bytes = _sysctl("hw.memsize")
    return {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": _sysctl("machdep.cpu.brand_string") or platform.processor(),
        "cpu_cores": int(_sysctl("hw.ncpu") or 0),
        "ram_gb": round(int(mem_bytes) / 1024**3, 1) if mem_bytes else None,
        "python": platform.python_version(),
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "device_used": "cpu",
        "packages": versions,
    }


def main() -> None:
    env = collect_environment()
    ENVIRONMENT_FILE.write_text(json.dumps(env, indent=2) + "\n")
    print(f"Wrote {ENVIRONMENT_FILE}")
    print(json.dumps(env, indent=2))


if __name__ == "__main__":
    main()
