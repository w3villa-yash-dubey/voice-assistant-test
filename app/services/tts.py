"""Goonj TTS only — BH-Builds/goonj-1-82M (Kokoro-82M finetune)."""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import re
import sys
import types
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from app.config import config
from app.services.hinglish_lexicon import HINDI_LEX

logger = logging.getLogger(__name__)

REPO_ID = "BH-Builds/goonj-1-82M"
GOONJ_SAMPLE_RATE = 24000
DEVANAGARI = re.compile(r"[ऀ-ॿ]")

PERSONAS = (
    "hi_atul",
    "hi_meera",
    "hi_ravi",
    "hi_shivani",
    "en_aman",
    "en_ananya",
    "en_arjun",
    "en_dev",
    "en_divya",
    "en_kabir",
    "en_nisha",
    "en_priya",
    "en_sameer",
    "en_tara",
    "bed_hindi",
)


def _load_kmodel_class():
    """Load kokoro.KModel without importing KPipeline (avoids spacy / misaki[en])."""
    spec = importlib.util.find_spec("kokoro")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError(
            "kokoro is not installed. Run: pip install kokoro misaki --no-deps"
        )
    pkg = Path(list(spec.submodule_search_locations)[0])

    # Register a lightweight package shell so relative imports inside model.py work.
    if "kokoro" not in sys.modules or not getattr(sys.modules["kokoro"], "__path__", None):
        placeholder = types.ModuleType("kokoro")
        placeholder.__path__ = [str(pkg)]  # type: ignore[attr-defined]
        placeholder.__file__ = str(pkg / "__init__.py")
        sys.modules["kokoro"] = placeholder

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
    return model_mod.KModel


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


def detect_lang(text: str) -> str:
    if DEVANAGARI.search(text):
        return "hi"
    words = [w.lower() for w in re.findall(r"[a-zA-Z']+", text)]
    if not words:
        return "en"
    hits = sum(1 for w in words if w in HINDI_LEX)
    return "hing" if hits / len(words) > 0.15 else "en"


def resolve_persona(name: str) -> str:
    key = name.strip()
    if key in PERSONAS:
        return key
    for prefix in ("hi_", "en_"):
        candidate = f"{prefix}{key}"
        if candidate in PERSONAS:
            return candidate
    raise ValueError(f"Unknown GOONJ_PERSONA '{name}'. Choose one of: {', '.join(PERSONAS)}")


class GoonjTTS:
    """Local Goonj-1-82M synthesizer (24 kHz → resampled PCM16)."""

    def __init__(self) -> None:
        self._model = None
        self._voice_pack = None
        self._device = "cpu"
        self._g2p_hi = None
        self._g2p_en = None
        self._lock = asyncio.Lock()
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return

        from misaki.espeak import EspeakG2P

        persona = resolve_persona(config.goonj_persona)
        token = config.hf_token or None
        cache = config.goonj_cache_dir or None

        logger.info("Downloading / loading Goonj weights from %s (persona=%s)", REPO_ID, persona)
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
        elif wanted in ("mps", "auto") and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            # MPS can be flaky for this model; prefer CPU unless explicitly requested.
            self._device = "mps" if wanted == "mps" else "cpu"
        else:
            self._device = "cpu"

        KModel = _load_kmodel_class()
        # kokoro>=0.7 signature: KModel(config=..., model=...)
        self._model = KModel(config=config_path, model=model_path).to(self._device).eval()
        self._voice_pack = torch.load(voice_path, map_location="cpu", weights_only=True)

        self._g2p_hi = EspeakG2P(language="hi")
        self._g2p_en = EspeakG2P(language="en-us")
        self._loaded = True
        logger.info("Goonj ready on %s", self._device)

    def _phonemize(self, text: str) -> str:
        assert self._g2p_hi and self._g2p_en and self._model
        lang = config.goonj_lang if config.goonj_lang != "auto" else detect_lang(text)

        if lang == "hi":
            phonemes = self._run_g2p(self._g2p_hi, text)
        elif lang == "en":
            phonemes = self._run_g2p(self._g2p_en, text)
        else:
            phonemes = self._phonemize_hinglish(text)

        # Keep only symbols the model vocab knows.
        cleaned = "".join(ch for ch in (phonemes or "") if ch in self._model.vocab)
        if not cleaned.strip():
            raise RuntimeError(f"G2P produced empty phonemes for text={text!r} lang={lang}")
        return cleaned

    @staticmethod
    def _run_g2p(g2p, text: str) -> str:
        """misaki EspeakG2P may return str or (phonemes, tokens)."""
        result = g2p(text)
        if isinstance(result, tuple):
            return result[0] or ""
        return result or ""

    def _phonemize_hinglish(self, text: str) -> str:
        assert self._g2p_hi and self._g2p_en
        parts: list[str] = []
        for token in re.findall(r"[A-Za-z']+|[ऀ-ॿ]+|[^A-Za-z'ऀ-ॿ]+", text):
            lower = token.lower()
            if lower in HINDI_LEX:
                parts.append(self._run_g2p(self._g2p_hi, HINDI_LEX[lower]))
            elif DEVANAGARI.search(token):
                parts.append(self._run_g2p(self._g2p_hi, token))
            elif re.fullmatch(r"[A-Za-z']+", token):
                parts.append(self._run_g2p(self._g2p_en, token))
            else:
                parts.append(token)
        return "".join(parts)

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
        async with self._lock:
            return await asyncio.to_thread(self._synthesize_sync, text, sample_rate)


_tts: GoonjTTS | None = None


def build_tts() -> GoonjTTS:
    """Shared singleton — do not reload 82M weights per WebRTC session."""
    global _tts
    if _tts is None:
        _tts = GoonjTTS()
    return _tts


def preload_tts() -> None:
    build_tts().load()
