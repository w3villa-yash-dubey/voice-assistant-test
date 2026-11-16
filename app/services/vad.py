"""Energy + zero-crossing VAD (no native deps — Jetson / Python 3.13 friendly)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VadResult:
    is_speech: bool
    speech_ms: int
    silence_ms: int
    should_finalize: bool
    should_barge_in: bool


class VoiceActivityDetector:
    def __init__(
        self,
        sample_rate: int = 16000,
        aggressiveness: int = 2,
        silence_ms_to_end: int = 700,
        min_speech_ms: int = 250,
        barge_in_ms: int = 200,
        frame_ms: int = 20,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_samples = int(sample_rate * frame_ms / 1000)
        self.frame_bytes = self.frame_samples * 2
        self.silence_ms_to_end = silence_ms_to_end
        self.min_speech_ms = min_speech_ms
        self.barge_in_ms = barge_in_ms

        # aggressiveness 0..3 → lower energy threshold = more sensitive
        levels = (500.0, 350.0, 220.0, 140.0)
        self.energy_threshold = levels[max(0, min(3, aggressiveness))]

        self._speech_ms = 0
        self._silence_ms = 0
        self._in_speech = False
        self._residual = b""

    def reset(self) -> None:
        self._speech_ms = 0
        self._silence_ms = 0
        self._in_speech = False
        self._residual = b""

    def _is_speech_frame(self, frame: bytes) -> bool:
        audio = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
        if audio.size == 0:
            return False
        rms = float(np.sqrt(np.mean(audio * audio)))
        # Zero-crossing rate helps reject steady noise / hum.
        signs = np.sign(audio)
        signs[signs == 0] = 1
        zcr = float(np.mean(signs[:-1] != signs[1:])) if audio.size > 1 else 0.0
        return rms >= self.energy_threshold and 0.01 < zcr < 0.35

    def feed(self, pcm16_mono: bytes, *, speaking: bool = False) -> list[VadResult]:
        data = self._residual + pcm16_mono
        results: list[VadResult] = []
        offset = 0
        while offset + self.frame_bytes <= len(data):
            frame = data[offset : offset + self.frame_bytes]
            offset += self.frame_bytes
            is_speech = self._is_speech_frame(frame)

            if is_speech:
                self._speech_ms += self.frame_ms
                self._silence_ms = 0
                if self._speech_ms >= self.min_speech_ms:
                    self._in_speech = True
            else:
                self._silence_ms += self.frame_ms
                if not self._in_speech:
                    self._speech_ms = 0

            should_finalize = (
                self._in_speech and self._silence_ms >= self.silence_ms_to_end
            )
            should_barge_in = (
                speaking and is_speech and self._speech_ms >= self.barge_in_ms
            )

            results.append(
                VadResult(
                    is_speech=is_speech,
                    speech_ms=self._speech_ms,
                    silence_ms=self._silence_ms,
                    should_finalize=should_finalize,
                    should_barge_in=should_barge_in,
                )
            )

            if should_finalize:
                self._speech_ms = 0
                self._silence_ms = 0
                self._in_speech = False

        self._residual = data[offset:]
        return results
