"""Mithi v2 Flash TTS — Oomni/mithi-v2-flash (local, 24 kHz).

Download once:
  python scripts/download_mithi.py
Then set MITHI_MODEL_DIR to that folder (must contain run.py + weights).
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import sys
import threading
from pathlib import Path

import numpy as np

from app.config import config

logger = logging.getLogger(__name__)

MITHI_SAMPLE_RATE = 24000
_DEFAULT_VOICES = (
    "satvik",
    "anirban",
    "nakul",
    "vasant",
    "siddhant",
    "charvi",
    "ahana",
    "vaidehi",
    "malini",
    "jahnavi",
)

_import_lock = threading.Lock()
_Mithi = None
_SR = MITHI_SAMPLE_RATE


def _resolve_model_dir() -> Path:
    raw = (config.mithi_model_dir or "").strip()
    if not raw:
        raise RuntimeError(
            "MITHI_MODEL_DIR is empty. Download the model first:\n"
            "  python scripts/download_mithi.py\n"
            "Then set MITHI_MODEL_DIR in .env to that folder."
        )
    path = Path(raw).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"MITHI_MODEL_DIR does not exist: {path}")
    if not (path / "run.py").is_file():
        raise FileNotFoundError(
            f"MITHI_MODEL_DIR missing run.py: {path}\n"
            "Re-run: python scripts/download_mithi.py"
        )
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"MITHI_MODEL_DIR missing config.json: {path}")
    return path


def _resolve_device() -> str | None:
    wanted = (config.mithi_device or "auto").strip().lower()
    if wanted in ("", "auto"):
        return None  # Mithi picks cuda if available else cpu
    if wanted in ("cuda", "cpu"):
        return wanted
    raise ValueError(f"MITHI_DEVICE must be auto|cuda|cpu, got {wanted!r}")


def _import_mithi(model_dir: Path):
    """Load Mithi class from the downloaded repo's run.py (self-contained)."""
    global _Mithi, _SR
    with _import_lock:
        if _Mithi is not None:
            return _Mithi, _SR

        run_path = model_dir / "run.py"
        # Ensure model dir is importable for any relative assets Mithi expects.
        model_str = str(model_dir)
        if model_str not in sys.path:
            sys.path.insert(0, model_str)

        spec = importlib.util.spec_from_file_location("mithi_v2_flash_run", run_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load Mithi run.py from {run_path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["mithi_v2_flash_run"] = mod
        spec.loader.exec_module(mod)

        if not hasattr(mod, "Mithi"):
            raise ImportError(f"{run_path} did not export Mithi")
        _Mithi = mod.Mithi
        _SR = int(getattr(mod, "SR", MITHI_SAMPLE_RATE))
        logger.info("Imported Mithi from %s (SR=%s)", run_path, _SR)
        return _Mithi, _SR


def _resample_pcm16(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    if src_rate == dst_rate or not pcm:
        return pcm
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    dst_len = int(len(audio) * dst_rate / src_rate)
    if dst_len <= 0:
        return b""
    x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=dst_len, endpoint=False)
    return np.clip(np.interp(x_new, x_old, audio), -32768, 32767).astype(np.int16).tobytes()


def _float_audio_to_pcm16(audio) -> bytes:
    wave = np.asarray(audio, dtype=np.float32).reshape(-1)
    if wave.size == 0:
        return b""
    return np.clip(wave * 32767.0, -32768, 32767).astype(np.int16).tobytes()


class MithiTTS:
    """Local Mithi v2 Flash synthesizer → PCM16 (optionally resampled)."""

    def __init__(self) -> None:
        self._engine = None
        self._sr = MITHI_SAMPLE_RATE
        self._device = "cpu"
        self._async_lock = asyncio.Lock()
        self._load_lock = threading.Lock()
        self._loaded = False

    def load(self) -> None:
        with self._load_lock:
            if self._loaded:
                return

            model_dir = _resolve_model_dir()
            device = _resolve_device()
            Mithi, sr = _import_mithi(model_dir)

            logger.info(
                "Loading Mithi from %s (lang=%s voice=%s device=%s)",
                model_dir,
                config.mithi_lang,
                config.mithi_voice,
                device or "auto",
            )
            self._engine = Mithi(str(model_dir), device=device)
            self._sr = sr
            self._device = str(getattr(self._engine, "dev", device or "auto"))
            self._loaded = True
            voices = getattr(self._engine, "voices", list(_DEFAULT_VOICES))
            logger.info(
                "Mithi ready on %s (voices=%s)",
                self._device,
                ",".join(voices) if isinstance(voices, (list, tuple)) else voices,
            )

    def _synthesize_sync(self, text: str, sample_rate: int) -> bytes:
        self.load()
        assert self._engine is not None

        cleaned = text.strip()
        if not cleaned:
            return b""

        audio = self._engine.synthesize(
            cleaned,
            lang=config.mithi_lang,
            voice=config.mithi_voice,
        )
        pcm = _float_audio_to_pcm16(audio)
        return _resample_pcm16(pcm, self._sr, sample_rate)

    async def synthesize(self, text: str, sample_rate: int = MITHI_SAMPLE_RATE) -> bytes:
        if not text.strip():
            return b""
        async with self._async_lock:
            return await asyncio.to_thread(self._synthesize_sync, text, sample_rate)


_tts: MithiTTS | None = None


def build_tts() -> MithiTTS:
    """Shared singleton — do not reload ~99M weights per session."""
    global _tts
    if _tts is None:
        _tts = MithiTTS()
    return _tts


def preload_tts() -> None:
    build_tts().load()
