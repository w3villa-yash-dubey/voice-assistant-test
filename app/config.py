"""Runtime configuration from environment."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()
load_dotenv(".env.local")


_DEFAULT_SYSTEM_PROMPT = """\
You are a real-time voice assistant on a LiveKit call.

Style:
- Sound natural, warm, and conversational — like a helpful friend on a phone call.
- Prefer Hindi or Hinglish when the user speaks Hindi; otherwise match their language.
- Keep every reply short: usually 1–2 spoken sentences, never more than 3.
- No markdown, bullets, numbered lists, code fences, or emoji.
- No stage directions or meta talk ("as an AI…", "sure, here's…").
- Prefer plain spoken words that TTS can pronounce cleanly.

Turn-taking:
- Answer the latest user utterance directly; do not ramble.
- If the user interrupts or changes topic, drop the old thread and follow them.
- Ask at most one short clarifying question when needed.
- For greetings, reply briefly and invite them to talk.

Goal: minimal latency feel — short, clear, speakable answers only.
"""


@dataclass(frozen=True)
class Config:
    # Small web UI (token endpoint + static page)
    ui_host: str = os.getenv("UI_HOST", "0.0.0.0")
    ui_port: int = int(os.getenv("UI_PORT", "8080"))

    # LiveKit (local server or Cloud)
    livekit_url: str = os.getenv("LIVEKIT_URL", "ws://127.0.0.1:7880")
    livekit_api_key: str = os.getenv("LIVEKIT_API_KEY", "devkey")
    livekit_api_secret: str = os.getenv("LIVEKIT_API_SECRET", "secret")
    agent_name: str = os.getenv("AGENT_NAME", "goonj-voice-agent")


    # OpenAI-compatible API (STT + LLM). Works with OpenAI, Groq, local vLLM, etc.
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    stt_model: str = os.getenv("STT_MODEL", "whisper-1")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")

    system_prompt: str = os.getenv("SYSTEM_PROMPT", _DEFAULT_SYSTEM_PROMPT)

    # Goonj TTS — https://huggingface.co/BH-Builds/goonj-1-82M (optional / legacy)
    goonj_persona: str = os.getenv("GOONJ_PERSONA", "hi_meera")
    goonj_lang: str = os.getenv("GOONJ_LANG", "auto")  # auto | hi | en | hing
    goonj_speed: float = float(os.getenv("GOONJ_SPEED", "1.0"))
    goonj_device: str = os.getenv("GOONJ_DEVICE", "auto")  # auto | cpu | cuda | mps
    goonj_cache_dir: str = os.getenv("GOONJ_CACHE_DIR", "")
    hf_token: str = os.getenv("HF_TOKEN", "")

    # Mithi v2 Flash TTS — https://huggingface.co/Oomni/mithi-v2-flash
    # Download: python scripts/download_mithi.py
    mithi_model_dir: str = os.getenv("MITHI_MODEL_DIR", "models/mithi-v2-flash")
    mithi_lang: str = os.getenv("MITHI_LANG", "en")  # hi en bn gu kn ml mr or pa ta te
    mithi_voice: str = os.getenv("MITHI_VOICE", "ahana")
    # auto | cuda | cpu  (auto = cuda if available)
    mithi_device: str = os.getenv("MITHI_DEVICE", "auto")



config = Config()
