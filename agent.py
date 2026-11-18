"""Simple LiveKit voice agent with local Mithi v2 Flash TTS."""

from __future__ import annotations

import logging
import os

from livekit import agents
from livekit.agents import Agent, AgentServer, AgentSession, JobContext, JobProcess
from livekit.plugins import openai, silero

from app.config import config
from app.plugins.mithi_tts import MithiTTSPlugin

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("voice-agent")

# Ensure LiveKit CLI picks up local-server credentials from config.
os.environ.setdefault("LIVEKIT_URL", config.livekit_url)
os.environ.setdefault("LIVEKIT_API_KEY", config.livekit_api_key)
os.environ.setdefault("LIVEKIT_API_SECRET", config.livekit_api_secret)


_ENGLISH_INSTRUCTIONS = """\
You are a real-time English voice assistant on a LiveKit call.

Style:
- Speak English only. Never reply in Hindi, Hinglish, or any other language.
- Sound natural, warm, and conversational.
- Keep every reply short: usually 1–2 spoken sentences, never more than 3.
- No markdown, bullets, numbered lists, code fences, or emoji.
- No stage directions or meta talk.
- Prefer plain spoken words that TTS can pronounce cleanly.

Turn-taking:
- Answer the latest user utterance directly; do not ramble.
- If the user interrupts or changes topic, drop the old thread and follow them.
- Ask at most one short clarifying question when needed.
- For greetings, reply briefly in English and invite them to talk.
"""


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=_ENGLISH_INSTRUCTIONS)


# Nano-friendly: one idle process, higher load threshold.
# Do not preload Mithi here — it OOMs/timeouts when many processes load torch.
server = AgentServer(
    num_idle_processes=1,
    load_threshold=0.95,
    port=8091,  # 8081 is used by the other LiveKit agent on this host
)


def _prewarm(proc: JobProcess) -> None:
    """Load only Silero VAD; Mithi loads lazily on first TTS call."""
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = _prewarm


@server.rtc_session(agent_name=config.agent_name)
async def entrypoint(ctx: JobContext) -> None:
    if not config.openai_api_key:
        logger.warning("OPENAI_API_KEY is empty — STT/LLM calls will fail until set")

    vad = ctx.proc.userdata.get("vad") or silero.VAD.load()

    session = AgentSession(
        vad=vad,
        stt=openai.STT(
            model=config.stt_model,
            api_key=config.openai_api_key or None,
            base_url=config.openai_base_url,
            language="en",
            detect_language=False,
            use_realtime=False,
        ),
        llm=openai.LLM(
            model=config.llm_model,
            api_key=config.openai_api_key or None,
            base_url=config.openai_base_url,
            temperature=0.55,
            max_completion_tokens=100,
        ),
        tts=MithiTTSPlugin(),
    )

    await session.start(agent=Assistant(), room=ctx.room)
    await session.generate_reply(
        instructions="Greet the user briefly in a warm voice and ask how you can help."
    )


if __name__ == "__main__":
    agents.cli.run_app(server)
