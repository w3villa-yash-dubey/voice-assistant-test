"""One WebRTC peer connection = one simple voice agent session."""

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
    """Minimal realtime loop: listen → STT → stream LLM → chunked TTS, with barge-in."""

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

    def _busy(self) -> bool:
        return self.state in (AgentState.THINKING, AgentState.SPEAKING)

    async def _consume_inbound(self, track: MediaStreamTrack) -> None:
        try:
            while True:
                frame: AudioFrame = await track.recv()
                pcm16 = _frame_to_pcm16_mono(frame, config.sample_rate)
                if not pcm16:
                    continue

                busy = self._busy()
                results = self.vad.feed(pcm16, speaking=busy)

                # Buffer speech while idle/listening, or while interrupting a reply.
                if self.state in (AgentState.IDLE, AgentState.LISTENING) or (
                    busy and any(r.is_speech for r in results)
                ):
                    self._utterance.extend(pcm16)

                for result in results:
                    if result.should_barge_in and busy:
                        logger.info("Barge-in detected (state=%s)", self.state.value)
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
        """Stop TTS / in-flight reply immediately; keep mic audio as the new utterance."""
        self._cancel_reply.set()
        self.outbound.clear()

        task = self._reply_task
        self._reply_task = None
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._cancel_reply.clear()
        self.state = AgentState.LISTENING
        # Keep residual speech in _utterance; only reset VAD counters for a fresh end-point.
        self.vad.reset()
        # Seed VAD as already in-speech so a short pause can finalize the interrupt.
        self.vad.mark_in_speech(config.min_speech_ms)
        await self._emit({"type": "state", "state": self.state.value, "barge_in": True})

    async def _run_turn(self, pcm_utterance: bytes) -> None:
        spoken_any = False
        reply_parts: list[str] = []
        try:
            await self._emit({"type": "status", "message": "transcribing"})
            text = await self.stt.transcribe(pcm_utterance, config.sample_rate)
            if self._cancel_reply.is_set():
                return
            if not text:
                self.state = AgentState.IDLE
                await self._emit({"type": "state", "state": self.state.value})
                return

            await self._emit({"type": "user_transcript", "text": text})
            await self._emit({"type": "status", "message": "thinking"})

            # Stream LLM → synthesize each sentence → enqueue ASAP for low TTFB.
            async for sentence in self.llm.reply_sentences(text):
                if self._cancel_reply.is_set():
                    return
                reply_parts.append(sentence)
                await self._emit(
                    {
                        "type": "assistant_text",
                        "text": sentence,
                        "partial": True,
                    }
                )

                pcm = await self.tts.synthesize(sentence, sample_rate=config.sample_rate)
                if self._cancel_reply.is_set():
                    return
                if not pcm:
                    continue

                if not spoken_any:
                    spoken_any = True
                    self.state = AgentState.SPEAKING
                    await self._emit({"type": "status", "message": "speaking"})
                    await self._emit({"type": "state", "state": self.state.value})

                await self.outbound.enqueue_pcm16(pcm, src_rate=config.sample_rate)

            full = " ".join(reply_parts).strip()
            if full:
                await self._emit({"type": "assistant_text", "text": full, "partial": False})

            if spoken_any and not self._cancel_reply.is_set():
                await self.outbound.wait_drained(self._cancel_reply)

            if not self._cancel_reply.is_set() and self.state == AgentState.SPEAKING:
                self.state = AgentState.IDLE
                self._utterance.clear()
                self.vad.reset()
                await self._emit({"type": "state", "state": self.state.value})
            elif not spoken_any and not self._cancel_reply.is_set():
                self.state = AgentState.IDLE
                await self._emit({"type": "state", "state": self.state.value})
        except asyncio.CancelledError:
            self.outbound.clear()
            raise
        except Exception as exc:
            logger.exception("Turn failed")
            self.outbound.clear()
            self.state = AgentState.IDLE
            await self._emit({"type": "error", "message": str(exc)})
            await self._emit({"type": "state", "state": self.state.value})

    async def close(self) -> None:
        self._cancel_reply.set()
        self.outbound.clear()
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
