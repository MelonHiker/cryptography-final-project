from __future__ import annotations

from typing import Sequence

import numpy as np


def _to_float_array(values: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        return array.reshape(-1)
    return array


def _safe_mode(values: Sequence[float] | np.ndarray) -> float:
    array = _to_float_array(values)
    if array.size == 0:
        return 0.0
    if array.size < 3:
        return float(array[0])

    counts, edges = np.histogram(array, bins="auto")
    max_bin_index = np.argmax(counts)
    return float((edges[max_bin_index] + edges[max_bin_index + 1]) / 2.0)


def _safe_stats(
    values: Sequence[float] | np.ndarray,
) -> tuple[float, float, float, float, float, float]:
    array = _to_float_array(values)
    if array.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=0))
    maximum = float(np.max(array))
    minimum = float(np.min(array))
    mode = _safe_mode(array)
    median = float(np.median(array))
    return mean, std, maximum, minimum, mode, median


def _next_power_of_two(length: int) -> int:
    if length <= 1:
        return 1
    return 1 << (length - 1).bit_length()


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (np.power(10.0, mel / 2595.0) - 1.0)


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_filter_bank(sample_rate: int, n_fft: int, num_filters: int) -> np.ndarray:
    low_mel = _hz_to_mel(np.array([0.0], dtype=np.float64))[0]
    high_mel = _hz_to_mel(np.array([sample_rate / 2.0], dtype=np.float64))[0]
    mel_points = np.linspace(low_mel, high_mel, num_filters + 2)
    hz_points = _mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    filters = np.zeros((num_filters, n_fft // 2 + 1), dtype=np.float64)
    for index in range(1, num_filters + 1):
        left = bin_points[index - 1]
        center = bin_points[index]
        right = bin_points[index + 1]

        if center > left:
            for bin_index in range(left, center):
                if 0 <= bin_index < filters.shape[1]:
                    filters[index - 1, bin_index] = (bin_index - left) / (center - left)

        if right > center:
            for bin_index in range(center, right):
                if 0 <= bin_index < filters.shape[1]:
                    filters[index - 1, bin_index] = (right - bin_index) / (right - center)

    return filters


def _dct_matrix(num_coeffs: int, num_filters: int) -> np.ndarray:
    basis = np.zeros((num_coeffs, num_filters), dtype=np.float64)
    scale = np.pi / num_filters
    factor = np.sqrt(2.0 / num_filters)
    for coeff in range(num_coeffs):
        basis[coeff] = factor * np.cos((np.arange(num_filters) + 0.5) * coeff * scale)
    basis[0] *= 1.0 / np.sqrt(2.0)
    return basis


def compute_mfcc(
    audio: Sequence[float] | np.ndarray,
    sample_rate: int,
    num_coeffs: int = 32,
    num_filters: int = 32,
) -> np.ndarray:
    signal = _to_float_array(audio)
    if signal.size == 0:
        return np.zeros(num_coeffs, dtype=np.float64)

    n_fft = _next_power_of_two(signal.size)
    spectrum = np.abs(np.fft.rfft(signal, n=n_fft)) ** 2
    filters = _mel_filter_bank(sample_rate, n_fft, num_filters)
    energies = filters @ spectrum
    energies = np.where(energies <= 0.0, np.finfo(np.float64).eps, energies)
    log_energies = np.log(energies)
    coeffs = _dct_matrix(num_coeffs, num_filters) @ log_energies
    return coeffs.astype(np.float64)


def align_to_peak(
    audio: Sequence[float] | np.ndarray,
    timestamp: float,
    window_size: float,
    sample_rate: int = 44100,
    key_length: int = 4500,
) -> np.ndarray:
    signal = _to_float_array(audio)
    if signal.size == 0:
        return np.zeros(0, dtype=np.float64)

    center_index = int(round(float(timestamp) * sample_rate))
    search_radius = max(int(round(float(window_size) * sample_rate)), 0)
    search_start = max(center_index - search_radius, 0)
    search_stop = min(center_index + search_radius + 1, signal.size)

    if search_start >= search_stop:
        peak_index = min(max(center_index, 0), signal.size - 1)
    else:
        search_window = signal[search_start:search_stop]
        peak_index = search_start + int(np.argmax(np.abs(search_window)))

    pre_peak_length = int(key_length * 0.05)
    actual_start = max(0, peak_index - pre_peak_length)
    stop_index = min(actual_start + max(int(key_length), 1), signal.size)

    segment = signal[actual_start:stop_index]
    if segment.size < key_length:
        segment = np.pad(segment, (0, key_length - segment.size), "constant")
    return segment


def _window_energy(audio: np.ndarray, center_index: int, window_length: int) -> float:
    half = window_length // 2
    start = max(center_index - half, 0)
    stop = min(center_index + half, audio.size)
    if start >= stop:
        return 0.0
    return float(np.sum(np.abs(audio[start:stop])))


def _histogram_features(values: np.ndarray, bins: int) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 0.0

    counts, edges = np.histogram(values, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2.0

    right_indices = np.argsort(centers)[-3:]
    right_counts = counts[right_indices]
    right_centers = centers[right_indices]
    right_total = float(np.sum(right_counts))
    msc = (
        float(np.sum(right_centers * right_counts) / right_total)
        if right_total
        else float(right_centers[-1])
    )

    top_indices = np.argsort(counts)[::-1][:3]
    top_counts = counts[top_indices]
    top_centers = centers[top_indices]
    top_total = float(np.sum(top_counts))
    mfc = (
        float(np.sum(top_centers * top_counts) / top_total) if top_total else float(top_centers[0])
    )

    return msc, mfc


def extract_46_features(
    audio: Sequence[float] | np.ndarray,
    timestamps_sec: Sequence[float] | np.ndarray,
    sample_rate: int,
    key_length: int,
    histogram_bins: int = 40,
    keysyms: Sequence[str] | None = None,
) -> np.ndarray:
    signal = _to_float_array(audio)
    timestamps = _to_float_array(timestamps_sec)

    # Trim the ambient noise/silence before the first keystroke
    if timestamps.size > 0:
        trim_sec = max(float(timestamps[0]) - 0.5, 0.0)
        trim_samples = int(trim_sec * sample_rate)
        if 0 < trim_samples < signal.size:
            signal = signal[trim_samples:]
            timestamps = timestamps - trim_sec

    if timestamps.size == 0:
        timestamps = np.zeros(0, dtype=np.float64)

    alignment_window_sec = 0.05
    aligned_segments = [
        align_to_peak(
            signal,
            timestamp,
            alignment_window_sec,
            sample_rate=sample_rate,
            key_length=key_length,
        )
        for timestamp in timestamps
    ]

    if aligned_segments:
        mfcc_matrix = np.vstack(
            [
                compute_mfcc(segment, sample_rate, num_coeffs=32, num_filters=32)
                for segment in aligned_segments
            ]
        )
        mfcc = np.mean(mfcc_matrix, axis=0)
        energies = np.array(
            [float(np.sum(np.abs(segment))) for segment in aligned_segments], dtype=np.float64
        )
    else:
        mfcc = np.zeros(32, dtype=np.float64)
        energies = np.zeros(0, dtype=np.float64)

    strength_mean, strength_std, strength_max, strength_min, strength_mode, strength_median = (
        _safe_stats(energies)
    )
    msc, mfc = _histogram_features(energies, histogram_bins)

    if timestamps.size >= 2:
        diffs = np.diff(timestamps)
        if keysyms is not None and len(keysyms) == timestamps.size:
            keep_diff = np.ones(len(diffs), dtype=bool)
            for i, sym in enumerate(keysyms):
                if sym == "BackSpace":
                    if i - 1 >= 0:
                        keep_diff[i - 1] = False
                    if i < len(diffs):
                        keep_diff[i] = False
            purified_diffs = diffs[keep_diff]
            diffs_for_stats = purified_diffs if purified_diffs.size > 0 else diffs
        else:
            diffs_for_stats = diffs
    else:
        diffs_for_stats = np.zeros(0, dtype=np.float64)

    total_time = float(timestamps[-1] - timestamps[0]) if timestamps.size >= 2 else 0.0
    if total_time > 0.0:
        diffs_for_stats = diffs_for_stats / total_time

    timing_mean, timing_std, timing_max, timing_min, timing_mode, timing_median = _safe_stats(
        diffs_for_stats
    )
    key_count = float(timestamps.size)

    features = np.concatenate(
        [
            mfcc,
            np.array(
                [
                    strength_mean,
                    strength_std,
                    strength_mode,
                    strength_median,
                    msc,
                    mfc,
                    timing_mean,
                    timing_std,
                    timing_max,
                    timing_min,
                    timing_mode,
                    timing_median,
                    key_count,
                    total_time,
                ],
                dtype=np.float64,
            ),
        ]
    )

    if features.size != 46:
        raise ValueError(f"Expected 46 features, got {features.size}")

    return features
