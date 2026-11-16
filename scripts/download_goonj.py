"""Prefetch BH-Builds/goonj-1-82M weights + selected persona voicepack."""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

load_dotenv()

REPO = "BH-Builds/goonj-1-82M"


def main() -> int:
    persona = os.getenv("GOONJ_PERSONA", "hi_meera")
    token = os.getenv("HF_TOKEN") or None
    cache = os.getenv("GOONJ_CACHE_DIR") or None

    files = [
        "kokoro_hindi_final.pth",
        "config.json",
        f"voices/{persona}.pt",
    ]
    for name in files:
        path = hf_hub_download(REPO, name, token=token, cache_dir=cache)
        print(f"ok {name} -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
