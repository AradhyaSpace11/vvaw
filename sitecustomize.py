"""Project-local runtime defaults for Windows-friendly command launches."""

from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / ".cache"

_ENV_DEFAULTS = {
    "YOLO_CONFIG_DIR": CACHE_DIR / "ultralytics",
    "HF_HOME": CACHE_DIR / "huggingface",
    "TORCH_HOME": CACHE_DIR / "torch",
    "MPLCONFIGDIR": CACHE_DIR / "matplotlib",
    "PIP_CACHE_DIR": CACHE_DIR / "pip",
}

for name, path in _ENV_DEFAULTS.items():
    path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(name, str(path))
