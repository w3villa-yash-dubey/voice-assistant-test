"""Chat LLM via OpenAI-compatible API — stream sentences for low-latency TTS."""

from __future__ import annotations

import re
from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from app.config import config

# Split once we have a spoken clause (., ?, !, or Hindi danda / ellipsis).
_SENTENCE_END = re.compile(r"(?<=[\.!\?\u0964…])\s+")


class ChatLLM:
    def __init__(self) -> None:
        self._client = AsyncOpenAI(
            api_key=config.openai_api_key or "missing",
            base_url=config.openai_base_url,
        )
        self._history: list[dict[str, str]] = [
            {"role": "system", "content": config.system_prompt}
        ]

    def reset(self) -> None:
        self._history = [{"role": "system", "content": config.system_prompt}]

    async def reply(self, user_text: str) -> str:
        parts: list[str] = []
        async for sentence in self.reply_sentences(user_text):
            parts.append(sentence)
        return " ".join(parts).strip()

    async def reply_sentences(self, user_text: str) -> AsyncIterator[str]:
        """Stream the model and yield complete speakable sentences as soon as ready."""
        self._history.append({"role": "user", "content": user_text})
        # Keep context small for Jetson / low latency.
        messages = [self._history[0], *self._history[1:][-6:]]

        stream = await self._client.chat.completions.create(
            model=config.llm_model,
            messages=messages,
            temperature=0.55,
            max_tokens=100,
            stream=True,
        )

        buf = ""
        full = ""
        async for event in stream:
            delta = ""
            if event.choices:
                delta = event.choices[0].delta.content or ""
            if not delta:
                continue
            buf += delta
            full += delta

            while True:
                match = _SENTENCE_END.search(buf)
                if not match:
                    break
                sentence = buf[: match.start()].strip()
                buf = buf[match.end() :]
                if sentence:
                    yield sentence

        tail = buf.strip()
        if tail:
            yield tail

        text = full.strip()
        self._history.append({"role": "assistant", "content": text})
