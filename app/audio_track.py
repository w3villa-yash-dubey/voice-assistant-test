"""Outbound WebRTC audio track that streams PCM16 from a queue."""

from __future__ import annotations

import asyncio
import fractions
import time

import numpy as np
from av import AudioFrame
from aiortc import MediaStreamTrack


class OutboundAudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, sample_rate: int = 48000) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._buffer = b""
        self._samples_per_frame = 960  # 20ms @ 48kHz
        self._pts = 0
        self._start = time.time()
        self._cancelled = False

    def clear(self) -> None:
        """Drop queued TTS audio (barge-in)."""
        self._cancelled = True
        self._buffer = b""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._cancelled = False

    async def enqueue_pcm16(self, pcm16_mono_16k: bytes, src_rate: int = 16000) -> None:
        if not pcm16_mono_16k:
            return
        pcm48 = _resample(pcm16_mono_16k, src_rate, self.sample_rate)
        await self._queue.put(pcm48)

    async def recv(self) -> AudioFrame:
        # Produce continuous 20ms frames (silence when idle).
        needed = self._samples_per_frame * 2
        while len(self._buffer) < needed:
            try:
                chunk = await asyncio.wait_for(self._queue.get(), timeout=0.02)
            except asyncio.TimeoutError:
                chunk = None
            if chunk is None:
                missing = needed - len(self._buffer)
                self._buffer += b"\x00" * missing
                break
            if self._cancelled:
                continue
            self._buffer += chunk

        frame_bytes = self._buffer[:needed]
        self._buffer = self._buffer[needed:]

        samples = np.frombuffer(frame_bytes, dtype=np.int16).reshape(1, -1)

        frame = AudioFrame.from_ndarray(samples, format="s16", layout="mono")
        frame.sample_rate = self.sample_rate
        frame.pts = self._pts
        frame.time_base = fractions.Fraction(1, self.sample_rate)
        self._pts += self._samples_per_frame

        # Pace roughly real-time so we don't flood the peer.
        elapsed = time.time() - self._start
        expected = self._pts / self.sample_rate
        if expected > elapsed:
            await asyncio.sleep(expected - elapsed)

        return frame


def _resample(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    if src_rate == dst_rate or not pcm:
        return pcm
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    duration = len(audio) / src_rate
    dst_len = max(1, int(duration * dst_rate))
    x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=dst_len, endpoint=False)
    out = np.interp(x_new, x_old, audio)
    return np.clip(out, -32768, 32767).astype(np.int16).tobytes()
