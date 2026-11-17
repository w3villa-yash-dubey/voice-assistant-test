"""Simple LiveKit voice agent with local Goonj TTS."""

from __future__ import annotations

import logging
import os

from livekit import agents
from livekit.agents import Agent, AgentServer, AgentSession, JobContext, JobProcess
from livekit.plugins import openai, silero

from app.config import config
from app.plugins.goonj_tts import GoonjTTSPlugin

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("voice-agent")

# Ensure LiveKit CLI picks up local-server credentials from config.
os.environ.setdefault("LIVEKIT_URL", config.livekit_url)
os.environ.setdefault("LIVEKIT_API_KEY", config.livekit_api_key)
os.environ.setdefault("LIVEKIT_API_SECRET", config.livekit_api_secret)


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=config.system_prompt)


# Nano-friendly: one idle process, higher load threshold.
# Do not preload Goonj here — it OOMs/timeouts when many processes load torch.
server = AgentServer(
    num_idle_processes=1,
    load_threshold=0.95,
)


def _prewarm(proc: JobProcess) -> None:
    """Load only Silero VAD; Goonj loads lazily on first TTS call."""
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = _prewarm


@server.rtc_session(agent_name="goonj-voice-agent")
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
            detect_language=True,
            use_realtime=False,
        ),
        llm=openai.LLM(
            model=config.llm_model,
            api_key=config.openai_api_key or None,
            base_url=config.openai_base_url,
            temperature=0.55,
            max_completion_tokens=100,
        ),
        tts=GoonjTTSPlugin(),
    )

    await session.start(agent=Assistant(), room=ctx.room)
    await session.generate_reply(
        instructions="Greet the user briefly in a warm voice and ask how you can help."
    )


if __name__ == "__main__":
    agents.cli.run_app(server)
