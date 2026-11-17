"""LiveKit TTS plugin wrapping local Mithi v2 Flash."""

from __future__ import annotations

import logging

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectOptions,
    tts,
    utils,
)

from app.services.mithi_tts import MITHI_SAMPLE_RATE, MithiTTS, build_tts, preload_tts

logger = logging.getLogger(__name__)

# Push ~20 ms PCM frames for smoother playback.
_FRAME_BYTES = int(MITHI_SAMPLE_RATE * 0.02) * 2  # int16 mono


class MithiTTSPlugin(tts.TTS):
    """Non-streaming local TTS backed by Mithi v2 Flash."""

    def __init__(self, engine: MithiTTS | None = None) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=MITHI_SAMPLE_RATE,
            num_channels=1,
        )
        self._engine = engine or build_tts()

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.ChunkedStream:
        return _MithiChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
        )


class _MithiChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        tts: MithiTTSPlugin,
        input_text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=self._tts.sample_rate,
            num_channels=self._tts.num_channels,
            mime_type="audio/pcm",
        )

        text = (self.input_text or "").strip()
        if not text:
            output_emitter.flush()
            return

        pcm = await self._tts._engine.synthesize(text, sample_rate=MITHI_SAMPLE_RATE)
        if not pcm:
            logger.warning("Mithi returned empty audio for text=%r", text)
            output_emitter.flush()
            return

        for i in range(0, len(pcm), _FRAME_BYTES):
            chunk = pcm[i : i + _FRAME_BYTES]
            if chunk:
                output_emitter.push(chunk)

        output_emitter.flush()


__all__ = ["MithiTTSPlugin", "preload_tts"]
