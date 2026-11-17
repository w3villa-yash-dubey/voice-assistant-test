"""Tiny UI server: static page + LiveKit token (dispatches the voice agent)."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from aiohttp import web
from livekit import api

from app.config import config

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("voice-ui")


async def index(_request: web.Request) -> web.Response:
    return web.FileResponse(STATIC / "index.html")


async def health(_request: web.Request) -> web.Response:
    return web.json_response(
        {
            "ok": True,
            "livekit_url": config.livekit_url,
            "agent_name": config.agent_name,
        }
    )


async def token(_request: web.Request) -> web.Response:
    """Issue a join token that dispatches goonj-voice-agent into a fresh room."""
    room = f"voice-{uuid.uuid4().hex[:10]}"
    identity = f"user-{uuid.uuid4().hex[:8]}"

    grant = api.VideoGrants(
        room_join=True,
        room=room,
        can_publish=True,
        can_subscribe=True,
        can_publish_data=True,
    )
    access = (
        api.AccessToken(config.livekit_api_key, config.livekit_api_secret)
        .with_identity(identity)
        .with_name("You")
        .with_grants(grant)
        .with_room_config(
            api.RoomConfiguration(
                agents=[api.RoomAgentDispatch(agent_name=config.agent_name)]
            )
        )
    )

    return web.json_response(
        {
            "token": access.to_jwt(),
            "url": config.livekit_url,
            "room": room,
            "identity": identity,
            "agent_name": config.agent_name,
        }
    )


def create_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/health", health)
    app.router.add_post("/api/token", token)
    app.router.add_get("/api/token", token)
    app.router.add_static("/static/", STATIC, show_index=False)
    return app


def main() -> None:
    if not STATIC.exists():
        raise SystemExit(f"Missing static UI folder: {STATIC}")

    logger.info(
        "UI on http://127.0.0.1:%s  (LiveKit=%s agent=%s)",
        config.ui_port,
        config.livekit_url,
        config.agent_name,
    )
    web.run_app(create_app(), host=config.ui_host, port=config.ui_port, print=None)


if __name__ == "__main__":
    main()
