"""One WebRTC peer connection = one voice agent session."""

from __future__ import annotations

import asyncio
import logging
from enum import Enum
from typing import Any, Callable, Awaitable

import numpy as np
from av import AudioFrame
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack

from app.audio_track import OutboundAudioTrack
from app.config import config
from app.services.llm import ChatLLM
from app.services.stt import SpeechToText
from app.services.tts import GoonjTTS
from app.services.vad import VoiceActivityDetector

logger = logging.getLogger(__name__)

EventCallback = Callable[[dict[str, Any]], Awaitable[None] | None]


class AgentState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


class AgentSession:
    def __init__(
        self,
        stt: SpeechToText,
        llm: ChatLLM,
        tts: GoonjTTS,
        on_event: EventCallback | None = None,
    ) -> None:
        self.pc = RTCPeerConnection()
        self.outbound = OutboundAudioTrack(sample_rate=48000)
        self.pc.addTrack(self.outbound)

        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.on_event = on_event

        self.state = AgentState.IDLE
        self.vad = VoiceActivityDetector(
            sample_rate=config.sample_rate,
            aggressiveness=config.vad_aggressiveness,
            silence_ms_to_end=config.silence_ms_to_end,
            min_speech_ms=config.min_speech_ms,
            barge_in_ms=config.barge_in_ms,
        )

        self._utterance = bytearray()
        self._recv_task: asyncio.Task | None = None
        self._reply_task: asyncio.Task | None = None
        self._cancel_reply = asyncio.Event()

        @self.pc.on("track")
        def on_track(track: MediaStreamTrack) -> None:
            if track.kind != "audio":
                return
            logger.info("Inbound audio track received")
            self._recv_task = asyncio.create_task(self._consume_inbound(track))

        @self.pc.on("connectionstatechange")
        async def on_state_change() -> None:
            logger.info("Peer connection state: %s", self.pc.connectionState)
            await self._emit({"type": "pc_state", "state": self.pc.connectionState})
            if self.pc.connectionState in ("failed", "closed", "disconnected"):
                await self.close()

    async def _emit(self, event: dict[str, Any]) -> None:
        if not self.on_event:
            return
        result = self.on_event(event)
        if asyncio.iscoroutine(result):
            await result

    async def handle_offer(self, sdp: str, type_: str = "offer") -> RTCSessionDescription:
        offer = RTCSessionDescription(sdp=sdp, type=type_)
        await self.pc.setRemoteDescription(offer)
        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)
        assert self.pc.localDescription is not None
        return self.pc.localDescription

    async def _consume_inbound(self, track: MediaStreamTrack) -> None:
        try:
            while True:
                frame: AudioFrame = await track.recv()
                pcm16 = _frame_to_pcm16_mono(frame, config.sample_rate)
                if not pcm16:
                    continue

                speaking = self.state == AgentState.SPEAKING
                results = self.vad.feed(pcm16, speaking=speaking)

                # Always buffer while we might be in an utterance.
                if self.state in (AgentState.IDLE, AgentState.LISTENING) or (
                    self.state == AgentState.SPEAKING
                    and any(r.is_speech for r in results)
                ):
                    self._utterance.extend(pcm16)

                for result in results:
                    if result.should_barge_in and self.state == AgentState.SPEAKING:
                        logger.info("Barge-in detected")
                        await self._handle_barge_in()
                        break

                    if result.is_speech and self.state == AgentState.IDLE:
                        self.state = AgentState.LISTENING
                        await self._emit({"type": "state", "state": self.state.value})

                    if result.should_finalize and self.state == AgentState.LISTENING:
                        audio = bytes(self._utterance)
                        self._utterance.clear()
                        self.vad.reset()
                        self.state = AgentState.THINKING
                        await self._emit({"type": "state", "state": self.state.value})
                        self._reply_task = asyncio.create_task(self._run_turn(audio))
                        break
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Inbound audio loop failed")

    async def _handle_barge_in(self) -> None:
        self._cancel_reply.set()
        self.outbound.clear()
        if self._reply_task and not self._reply_task.done():
            self._reply_task.cancel()
            try:
                await self._reply_task
            except asyncio.CancelledError:
                pass
        self._cancel_reply.clear()
        # Keep recent audio as start of the new utterance.
        self.state = AgentState.LISTENING
        self.vad.reset()
        await self._emit({"type": "state", "state": self.state.value, "barge_in": True})

    async def _run_turn(self, pcm_utterance: bytes) -> None:
        try:
            await self._emit({"type": "status", "message": "transcribing"})
            text = await self.stt.transcribe(pcm_utterance, config.sample_rate)
            if not text:
                self.state = AgentState.IDLE
                await self._emit({"type": "state", "state": self.state.value})
                return

            await self._emit({"type": "user_transcript", "text": text})
            await self._emit({"type": "status", "message": "thinking"})
            reply = await self.llm.reply(text)
            await self._emit({"type": "assistant_text", "text": reply})

            if self._cancel_reply.is_set():
                return

            await self._emit({"type": "status", "message": "speaking"})
            self.state = AgentState.SPEAKING
            await self._emit({"type": "state", "state": self.state.value})

            pcm = await self.tts.synthesize(reply, sample_rate=config.sample_rate)
            if self._cancel_reply.is_set():
                return
            await self.outbound.enqueue_pcm16(pcm, src_rate=config.sample_rate)

            # Rough wait for playback to finish unless barge-in cancels.
            duration = len(pcm) / 2 / config.sample_rate
            await asyncio.sleep(duration + 0.15)

            if not self._cancel_reply.is_set() and self.state == AgentState.SPEAKING:
                self.state = AgentState.IDLE
                self._utterance.clear()
                self.vad.reset()
                await self._emit({"type": "state", "state": self.state.value})
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Turn failed")
            self.state = AgentState.IDLE
            await self._emit({"type": "error", "message": str(exc)})
            await self._emit({"type": "state", "state": self.state.value})

    async def close(self) -> None:
        if self._recv_task:
            self._recv_task.cancel()
        if self._reply_task:
            self._reply_task.cancel()
        await self.pc.close()


def _frame_to_pcm16_mono(frame: AudioFrame, target_rate: int) -> bytes:
    array = frame.to_ndarray()
    # Possible shapes: (channels, samples) or (1, channels*samples)
    if array.ndim == 2:
        if array.shape[0] <= 8:
            # (channels, samples)
            mono = array.mean(axis=0)
        else:
            mono = array.mean(axis=1)
    else:
        mono = array

    if frame.format.name.startswith("fl"):
        mono = np.clip(mono, -1.0, 1.0)
        pcm = (mono * 32767.0).astype(np.int16)
    else:
        pcm = mono.astype(np.int16)

    src_rate = frame.sample_rate
    if src_rate == target_rate:
        return pcm.tobytes()

    audio = pcm.astype(np.float32)
    dst_len = max(1, int(len(audio) * target_rate / src_rate))
    x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=dst_len, endpoint=False)
    out = np.interp(x_new, x_old, audio)
    return np.clip(out, -32768, 32767).astype(np.int16).tobytes()
