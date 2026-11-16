"""Runtime configuration from environment."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8080"))

    # OpenAI-compatible API (STT + LLM). Works with OpenAI, Groq, local vLLM, etc.
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    stt_model: str = os.getenv("STT_MODEL", "whisper-1")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")

    system_prompt: str = os.getenv(
        "SYSTEM_PROMPT",
        "You are a helpful voice assistant. Keep replies short and conversational "
        "(1-3 sentences). No markdown, no bullet lists. Prefer Hindi or Hinglish "
        "when the user speaks Hindi.",
    )

    # Goonj TTS — https://huggingface.co/BH-Builds/goonj-1-82M
    goonj_persona: str = os.getenv("GOONJ_PERSONA", "hi_meera")
    goonj_lang: str = os.getenv("GOONJ_LANG", "auto")  # auto | hi | en | hing
    goonj_speed: float = float(os.getenv("GOONJ_SPEED", "1.0"))
    goonj_device: str = os.getenv("GOONJ_DEVICE", "auto")  # auto | cpu | cuda | mps
    goonj_cache_dir: str = os.getenv("GOONJ_CACHE_DIR", "")
    hf_token: str = os.getenv("HF_TOKEN", "")

    # VAD / turn-taking
    vad_aggressiveness: int = int(os.getenv("VAD_AGGRESSIVENESS", "2"))
    silence_ms_to_end: int = int(os.getenv("SILENCE_MS_TO_END", "700"))
    min_speech_ms: int = int(os.getenv("MIN_SPEECH_MS", "250"))
    barge_in_ms: int = int(os.getenv("BARGE_IN_MS", "200"))

    sample_rate: int = 16000


config = Config()
