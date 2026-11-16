"""WebRTC voice assistant server (signaling + static UI)."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from aiohttp import WSMsgType, web

from app.config import config
from app.session import AgentSession
from app.services.llm import ChatLLM
from app.services.stt import SpeechToText
from app.services.tts import build_tts, preload_tts

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "static"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("voice-assistant")


async def index(_request: web.Request) -> web.Response:
    return web.FileResponse(STATIC / "index.html")


async def health(_request: web.Request) -> web.Response:
    return web.json_response(
        {
            "ok": True,
            "tts": "goonj-1-82M",
            "persona": config.goonj_persona,
            "stt_model": config.stt_model,
            "llm_model": config.llm_model,
        }
    )


async def websocket_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(heartbeat=20)
    await ws.prepare(request)
    logger.info("Client connected")

    stt = SpeechToText()
    llm = ChatLLM()
    tts = build_tts()

    async def on_event(event: dict) -> None:
        if not ws.closed:
            await ws.send_json(event)

    session = AgentSession(stt=stt, llm=llm, tts=tts, on_event=on_event)

    try:
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                if msg.type in (WSMsgType.CLOSE, WSMsgType.ERROR):
                    break
                continue

            try:
                data = json.loads(msg.data)
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict):
                continue

            msg_type = data.get("type")
            if msg_type == "offer":
                answer = await session.handle_offer(data["sdp"], data.get("sdpType", "offer"))
                await ws.send_json({"type": "answer", "sdp": answer.sdp, "sdpType": answer.type})
            elif msg_type == "reset":
                llm.reset()
                await ws.send_json({"type": "status", "message": "history cleared"})
            else:
                await ws.send_json({"type": "error", "message": f"unknown type: {msg_type}"})
    finally:
        await session.close()
        logger.info("Client disconnected")

    return ws


def create_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/health", health)
    app.router.add_get("/ws", websocket_handler)
    app.router.add_static("/static/", STATIC, show_index=False)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="WebRTC voice assistant")
    parser.add_argument("--host", default=config.host)
    parser.add_argument("--port", type=int, default=config.port)
    args = parser.parse_args()

    if not config.openai_api_key:
        logger.warning("OPENAI_API_KEY is empty — STT/LLM calls will fail until set")

    logger.info(
        "Loading Goonj TTS (persona=%s)…",
        config.goonj_persona,
    )
    try:
        preload_tts()
    except Exception:
        logger.exception(
            "Goonj preload failed — server will still start; first speak may retry load"
        )

    logger.info(
        "Starting on http://%s:%s (tts=goonj persona=%s)",
        args.host,
        args.port,
        config.goonj_persona,
    )
    web.run_app(create_app(), host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
