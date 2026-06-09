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
    signal: np.ndarray,
    peak_index: int,
    noise_floor: float,
    quiet_run: int = 75,
    max_radius: int | None = None,
) -> tuple[int, int]:
    peak_amplitude = float(signal[peak_index]) if signal.size else 0.0
    threshold = max(noise_floor * 3.0, float(np.percentile(signal, 65)), peak_amplitude * 0.08)
    min_left = max(peak_index - max_radius, 0) if max_radius is not None else 0
    max_right = min(peak_index + max_radius, signal.size - 1) if max_radius is not None else signal.size - 1

    left = peak_index
    quiet = 0
    while left > min_left:
        left -= 1
        if signal[left] <= threshold:
            quiet += 1
            if quiet >= quiet_run:
                left += quiet_run
                break
        else:
            quiet = 0
    else:
        if quiet > 0:
            left += quiet

    right = peak_index
    quiet = 0
    while right < max_right:
        right += 1
        if signal[right] <= threshold:
            quiet += 1
            if quiet >= quiet_run:
                right -= quiet_run
                break
        else:
            quiet = 0
    else:
        if quiet > 0:
            right -= quiet

    return max(left, 0), min(right, signal.size - 1)


def _robust_key_length(lengths: Sequence[int], sample_rate: int) -> int:
    if not lengths:
        return 0

    values = np.asarray(lengths, dtype=np.float64)
    min_length = max(int(round(sample_rate * 0.008)), 1)
    max_length = max(int(round(sample_rate * 0.12)), min_length)
    values = values[(values >= min_length) & (values <= max_length)]
    if values.size == 0:
        values = np.asarray(lengths, dtype=np.float64)

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad > 0:
        values = values[np.abs(values - median) <= 2.5 * mad]
    elif values.size > 2:
        values = values[values == median]

    if values.size == 0:
        values = np.asarray(lengths, dtype=np.float64)
    return int(round(float(np.median(values))))


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

    quiet_run = max(int(round(sample_rate * 0.002)), 50)
    max_radius = max(int(round(sample_rate * 0.08)), 1)
    for peak_index in peak_indices:
        left, right = _expand_key_bounds(
            signal,
            peak_index,
            noise_floor=noise_floor,
            quiet_run=quiet_run,
            max_radius=max_radius,
        )
        lengths.append(max(right - left + 1, 1))

    key_length = _robust_key_length(lengths, sample_rate)
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
