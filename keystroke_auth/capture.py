from __future__ import annotations

from dataclasses import dataclass
from threading import Event, Thread
from time import perf_counter, sleep

import numpy as np


class DependencyError(RuntimeError):
    pass


@dataclass(slots=True)
class TextCaptureResult:
    text: str
    timestamps_sec: list[float]
    invalidated: bool


def _import_pyaudio():
    try:
        import pyaudio  # type: ignore
    except ImportError as exc:
        raise DependencyError("pyaudio is required for live audio capture") from exc
    return pyaudio


def _import_keyboard():
    try:
        from pynput import keyboard  # type: ignore
    except ImportError as exc:
        raise DependencyError("pynput is required for keyboard capture") from exc
    return keyboard


class AudioRecorder:
    def __init__(self, sample_rate: int = 44100, chunk_size: int = 1024) -> None:
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size

    def record_until(self, stop_event: Event) -> np.ndarray:
        pyaudio = _import_pyaudio()
        audio_interface = pyaudio.PyAudio()
        stream = audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
        )

        frames: list[np.ndarray] = []
        try:
            while not stop_event.is_set():
                chunk = stream.read(self.chunk_size, exception_on_overflow=False)
                frames.append(np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0)
        finally:
            stream.stop_stream()
            stream.close()
            audio_interface.terminate()

        if not frames:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(frames)

    def record_fixed_duration(self, duration_sec: float) -> np.ndarray:
        stop_event = Event()
        samples_holder: dict[str, np.ndarray] = {}

        def _worker() -> None:
            samples_holder["samples"] = self.record_until(stop_event)

        thread = Thread(target=_worker, daemon=True)
        thread.start()
        sleep(duration_sec)
        stop_event.set()
        thread.join()
        return samples_holder.get("samples", np.zeros(0, dtype=np.float32))


def capture_space_presses(press_target: int = 3, timeout_sec: float = 20.0) -> list[float]:
    keyboard = _import_keyboard()
    timestamps: list[float] = []
    started = perf_counter()

    def on_press(key) -> bool | None:
        if key == keyboard.Key.space:
            timestamps.append(perf_counter() - started)
            if len(timestamps) >= press_target:
                return False
        return None

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    listener.join(timeout=timeout_sec)
    listener.stop()
    return timestamps[:press_target]


def capture_exact_text(target_text: str, timeout_sec: float = 30.0) -> TextCaptureResult:
    keyboard = _import_keyboard()
    text: list[str] = []
    timestamps: list[float] = []
    invalidated = False
    started = perf_counter()

    def on_press(key) -> bool | None:
        nonlocal invalidated
        if key == keyboard.Key.backspace:
            invalidated = True
            text.clear()
            timestamps.clear()
            return False

        try:
            char = key.char
        except AttributeError:
            return None

        if not char:
            return None

        text.append(char)
        timestamps.append(perf_counter() - started)

        current = "".join(text)
        if not target_text.startswith(current):
            invalidated = True
            text.clear()
            timestamps.clear()
            return False

        if current == target_text:
            return False

        return None

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    listener.join(timeout=timeout_sec)
    listener.stop()
    final_text = "".join(text)
    return TextCaptureResult(
        text=final_text,
        timestamps_sec=timestamps,
        invalidated=invalidated or final_text != target_text,
    )


def collect_calibration_capture(sample_rate: int, press_target: int = 3, tail_sec: float = 0.75):
    recorder = AudioRecorder(sample_rate=sample_rate)
    stop_event = Event()
    audio_holder: dict[str, np.ndarray] = {}

    def worker() -> None:
        audio_holder["audio"] = recorder.record_until(stop_event)

    thread = Thread(target=worker, daemon=True)
    thread.start()
    timestamps = capture_space_presses(press_target=press_target)
    if len(timestamps) < press_target:
        stop_event.set()
        thread.join()
        raise ValueError(f"Calibration needs {press_target} space presses, got {len(timestamps)}")
    sleep(tail_sec)
    stop_event.set()
    thread.join()
    return audio_holder.get("audio", np.zeros(0, dtype=np.float32)), timestamps


def collect_enrollment_capture(sample_rate: int, target_text: str, tail_sec: float = 0.75):
    recorder = AudioRecorder(sample_rate=sample_rate)
    stop_event = Event()
    audio_holder: dict[str, np.ndarray] = {}

    def worker() -> None:
        audio_holder["audio"] = recorder.record_until(stop_event)

    thread = Thread(target=worker, daemon=True)
    thread.start()
    text_result = capture_exact_text(target_text)
    sleep(tail_sec)
    stop_event.set()
    thread.join()
    return audio_holder.get("audio", np.zeros(0, dtype=np.float32)), text_result
