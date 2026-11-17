"""Download Oomni/mithi-v2-flash into MITHI_MODEL_DIR (default: models/mithi-v2-flash)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

load_dotenv()

REPO = "Oomni/mithi-v2-flash"


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    dest = Path(
        os.getenv("MITHI_MODEL_DIR") or (root / "models" / "mithi-v2-flash")
    ).expanduser()
    if not dest.is_absolute():
        dest = (root / dest).resolve()
    else:
        dest = dest.resolve()

    token = os.getenv("HF_TOKEN") or None
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {REPO} -> {dest}")
    path = snapshot_download(
        REPO,
        local_dir=str(dest),
        token=token,
    )
    run_py = Path(path) / "run.py"
    if not run_py.is_file():
        print(f"ERROR: expected run.py at {run_py}", file=sys.stderr)
        return 1

    print(f"ok model ready at {path}")
    print("Set in .env:")
    print(f"  MITHI_MODEL_DIR={path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
