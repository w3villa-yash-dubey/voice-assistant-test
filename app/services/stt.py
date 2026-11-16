"""Speech-to-text via OpenAI-compatible /audio/transcriptions."""

from __future__ import annotations

import io
import wave

from openai import AsyncOpenAI

from app.config import config


def _pcm16_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


class SpeechToText:
    def __init__(self) -> None:
        self._client = AsyncOpenAI(
            api_key=config.openai_api_key or "missing",
            base_url=config.openai_base_url,
        )

    async def transcribe(self, pcm16_mono: bytes, sample_rate: int = 16000) -> str:
        if not pcm16_mono:
            return ""
        wav = _pcm16_to_wav(pcm16_mono, sample_rate)
        result = await self._client.audio.transcriptions.create(
            model=config.stt_model,
            file=("speech.wav", wav, "audio/wav"),
        )
        return (result.text or "").strip()
