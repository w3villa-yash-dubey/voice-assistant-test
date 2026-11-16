"""Chat LLM via OpenAI-compatible API."""

from __future__ import annotations

from openai import AsyncOpenAI

from app.config import config


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
        self._history.append({"role": "user", "content": user_text})
        # Keep context small for Jetson / low latency.
        messages = [self._history[0], *self._history[1:][-8:]]
        result = await self._client.chat.completions.create(
            model=config.llm_model,
            messages=messages,
            temperature=0.6,
            max_tokens=120,
        )
        text = (result.choices[0].message.content or "").strip()
        self._history.append({"role": "assistant", "content": text})
        return text
