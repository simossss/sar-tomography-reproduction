"""Resolve the local SAR data root. Does not download or invent data.

Search order:
1. SAR_DATA_DIR
2. DATA_DIR
3. <repo>/data
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def data_dir() -> str:
    env = os.environ.get("SAR_DATA_DIR") or os.environ.get("DATA_DIR")
    if env:
        return os.path.abspath(env)
    return str(REPO_ROOT / "data")
