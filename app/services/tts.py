"""Goonj TTS only — BH-Builds/goonj-1-82M (Kokoro-82M finetune).

Hardcoded for this deployment: English + persona en_priya.
"""

from __future__ import annotations

import asyncio
import importlib.machinery
import importlib.util
import logging
import sys
import threading
import types
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from app.config import config

logger = logging.getLogger(__name__)

REPO_ID = "BH-Builds/goonj-1-82M"
GOONJ_SAMPLE_RATE = 24000

# Hardcoded — do not read persona/lang from env for this setup.
HARDCODED_PERSONA = "en_priya"
HARDCODED_LANG = "en"

_kmodel_lock = threading.Lock()
_KModel = None


def _kokoro_package_dir() -> Path:
    """Resolve installed kokoro package path without relying on a broken sys.modules stub."""
    # Drop incomplete placeholder that makes find_spec raise ValueError.
    existing = sys.modules.get("kokoro")
    if existing is not None and getattr(existing, "__spec__", None) is None:
        paths = list(getattr(existing, "__path__", []) or [])
        if paths:
            return Path(paths[0])
        for key in list(sys.modules):
            if key == "kokoro" or key.startswith("kokoro."):
                del sys.modules[key]

    try:
        spec = importlib.util.find_spec("kokoro")
    except ValueError:
        for key in list(sys.modules):
            if key == "kokoro" or key.startswith("kokoro."):
                del sys.modules[key]
        spec = importlib.util.find_spec("kokoro")

    if spec is None or not spec.submodule_search_locations:
        raise ImportError(
            "kokoro is not installed. Run: pip install kokoro misaki --no-deps"
        )
    return Path(list(spec.submodule_search_locations)[0])


def _load_kmodel_class():
    """Load kokoro.KModel without importing KPipeline (avoids spacy / misaki[en])."""
    global _KModel
    with _kmodel_lock:
        if _KModel is not None:
            return _KModel

        model_mod = sys.modules.get("kokoro.model")
        if model_mod is not None and hasattr(model_mod, "KModel"):
            _KModel = model_mod.KModel
            return _KModel

        pkg = _kokoro_package_dir()

        # Register a lightweight package shell so relative imports inside model.py work.
        if "kokoro" not in sys.modules or not getattr(sys.modules["kokoro"], "__path__", None):
            placeholder = types.ModuleType("kokoro")
            placeholder.__path__ = [str(pkg)]  # type: ignore[attr-defined]
            placeholder.__file__ = str(pkg / "__init__.py")
            placeholder.__spec__ = importlib.machinery.ModuleSpec(
                name="kokoro",
                loader=None,
                is_package=True,
            )
            placeholder.__spec__.submodule_search_locations = [str(pkg)]
            sys.modules["kokoro"] = placeholder
        else:
            # Ensure stub always has a real __spec__ so later find_spec calls succeed.
            mod = sys.modules["kokoro"]
            if getattr(mod, "__spec__", None) is None:
                mod.__spec__ = importlib.machinery.ModuleSpec(
                    name="kokoro",
                    loader=None,
                    is_package=True,
                )
                mod.__spec__.submodule_search_locations = list(
                    getattr(mod, "__path__", [str(pkg)])
                )

        def _load(mod_name: str, file_name: str):
            if mod_name in sys.modules and getattr(sys.modules[mod_name], "__file__", None):
                return sys.modules[mod_name]
            path = pkg / file_name
            mod_spec = importlib.util.spec_from_file_location(
                mod_name, path, submodule_search_locations=[str(pkg)]
            )
            assert mod_spec and mod_spec.loader
            mod = importlib.util.module_from_spec(mod_spec)
            sys.modules[mod_name] = mod
            mod_spec.loader.exec_module(mod)
            return mod

        _load("kokoro.istftnet", "istftnet.py")
        _load("kokoro.modules", "modules.py")
        model_mod = _load("kokoro.model", "model.py")
        _KModel = model_mod.KModel
        return _KModel


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


class GoonjTTS:
    """Local Goonj-1-82M synthesizer (24 kHz → resampled PCM16). English / en_priya only."""

    def __init__(self) -> None:
        self._model = None
        self._voice_pack = None
        self._device = "cpu"
        self._g2p_en = None
        self._async_lock = asyncio.Lock()
        self._load_lock = threading.Lock()
        self._loaded = False

    def load(self) -> None:
        with self._load_lock:
            if self._loaded:
                return

            from misaki.espeak import EspeakG2P

            persona = HARDCODED_PERSONA
            token = config.hf_token or None
            cache = config.goonj_cache_dir or None

            logger.info(
                "Loading Goonj weights from %s (persona=%s lang=%s)",
                REPO_ID,
                persona,
                HARDCODED_LANG,
            )
            model_path = hf_hub_download(
                REPO_ID, "kokoro_hindi_final.pth", token=token, cache_dir=cache
            )
            config_path = hf_hub_download(REPO_ID, "config.json", token=token, cache_dir=cache)
            voice_path = hf_hub_download(
                REPO_ID, f"voices/{persona}.pt", token=token, cache_dir=cache
            )

            wanted = (config.goonj_device or "auto").lower()
            if wanted in ("cuda", "auto") and torch.cuda.is_available():
                self._device = "cuda"
            elif (
                wanted in ("mps", "auto")
                and getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()
            ):
                self._device = "mps" if wanted == "mps" else "cpu"
            else:
                self._device = "cpu"

            KModel = _load_kmodel_class()
            self._model = KModel(config=config_path, model=model_path).to(self._device).eval()
            self._voice_pack = torch.load(voice_path, map_location="cpu", weights_only=True)

            self._g2p_en = EspeakG2P(language="en-us")
            self._loaded = True
            logger.info("Goonj ready on %s (en_priya / English only)", self._device)

    def _phonemize(self, text: str) -> str:
        assert self._g2p_en and self._model
        phonemes = self._run_g2p(self._g2p_en, text)
        cleaned = "".join(ch for ch in (phonemes or "") if ch in self._model.vocab)
        if not cleaned.strip():
            raise RuntimeError(f"G2P produced empty phonemes for text={text!r}")
        return cleaned

    @staticmethod
    def _run_g2p(g2p, text: str) -> str:
        """misaki EspeakG2P may return str or (phonemes, tokens)."""
        result = g2p(text)
        if isinstance(result, tuple):
            return result[0] or ""
        return result or ""

    def _synthesize_sync(self, text: str, sample_rate: int) -> bytes:
        self.load()
        assert self._model is not None and self._voice_pack is not None

        phonemes = self._phonemize(text.strip())
        n_ids = sum(1 for p in phonemes if p in self._model.vocab)
        idx = min(max(n_ids - 1, 0), self._voice_pack.shape[0] - 1)
        ref_s = self._voice_pack[idx].to(self._device)

        with torch.no_grad():
            audio = self._model(phonemes, ref_s, speed=config.goonj_speed)

        if hasattr(audio, "audio"):
            audio = audio.audio
        wave = audio.detach().float().cpu().numpy().reshape(-1)
        pcm = np.clip(wave * 32767.0, -32768, 32767).astype(np.int16).tobytes()
        return _resample_pcm16(pcm, GOONJ_SAMPLE_RATE, sample_rate)

    async def synthesize(self, text: str, sample_rate: int = 16000) -> bytes:
        if not text.strip():
            return b""
        async with self._async_lock:
            return await asyncio.to_thread(self._synthesize_sync, text, sample_rate)


_tts: GoonjTTS | None = None


def build_tts() -> GoonjTTS:
    """Shared singleton — do not reload 82M weights per session."""
    global _tts
    if _tts is None:
        _tts = GoonjTTS()
    return _tts


def preload_tts() -> None:
    build_tts().load()
