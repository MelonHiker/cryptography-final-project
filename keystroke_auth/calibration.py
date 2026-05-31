from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import AppConfig, save_config


@dataclass(slots=True)
class CalibrationResult:
    key_length: int
    sample_rate: int
    per_press_lengths: list[int]


def _first_significant_peak(signal: np.ndarray, around_index: int, search_radius: int) -> int:
    start = max(around_index - search_radius, 0)
    stop = min(around_index + search_radius + 1, signal.size)
    if start >= stop:
        return int(np.argmax(signal)) if signal.size else 0
    local = signal[start:stop]
    return int(start + np.argmax(local))


def _expand_key_bounds(
    signal: np.ndarray, peak_index: int, noise_floor: float, quiet_run: int = 75
) -> tuple[int, int]:
    threshold = max(noise_floor * 1.5, float(np.percentile(signal, 35)))

    left = peak_index
    quiet = 0
    while left > 0:
        left -= 1
        if signal[left] <= threshold:
            quiet += 1
            if quiet >= quiet_run:
                left += quiet_run
                break
        else:
            quiet = 0

    right = peak_index
    quiet = 0
    while right < signal.size - 1:
        right += 1
        if signal[right] <= threshold:
            quiet += 1
            if quiet >= quiet_run:
                right -= quiet_run
                break
        else:
            quiet = 0

    return max(left, 0), min(right, signal.size - 1)


def estimate_key_length(
    audio: Sequence[float] | np.ndarray,
    sample_rate: int,
    peak_hints_sec: Sequence[float] | np.ndarray | None = None,
) -> CalibrationResult:
    signal = np.abs(np.asarray(audio, dtype=np.float64).reshape(-1))
    if signal.size == 0:
        return CalibrationResult(key_length=0, sample_rate=sample_rate, per_press_lengths=[])

    noise_floor = float(np.median(signal))
    lengths: list[int] = []
    hints = (
        np.asarray(peak_hints_sec, dtype=np.float64).reshape(-1)
        if peak_hints_sec is not None
        else np.zeros(0)
    )

    if hints.size:
        peak_indices = [
            _first_significant_peak(
                signal, int(round(hint * sample_rate)), search_radius=max(sample_rate // 20, 1)
            )
            for hint in hints
        ]
    else:
        peak_indices = [int(np.argmax(signal))]

    for peak_index in peak_indices[:3]:
        left, right = _expand_key_bounds(signal, peak_index, noise_floor=noise_floor)
        lengths.append(max(right - left + 1, 1))

    key_length = int(round(float(np.mean(lengths)))) if lengths else 0
    return CalibrationResult(
        key_length=key_length, sample_rate=sample_rate, per_press_lengths=lengths
    )


def persist_calibration(
    config: AppConfig, calibration: CalibrationResult, path: str | Path = "config.json"
) -> AppConfig:
    config.sample_rate = calibration.sample_rate
    config.key_length = calibration.key_length
    config.calibrated = True
    save_config(config, path)
    return config
