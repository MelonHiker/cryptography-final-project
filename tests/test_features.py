"""Unit tests for the feature-extraction pipeline (features.py)."""

from __future__ import annotations

import numpy as np
import pytest

from keystroke_auth.features import (
    compute_mfcc,
    extract_46_features,
)

SAMPLE_RATE = 44100
KEY_LENGTH = 4500


def _toy_signal(n_keys: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    timestamps = np.linspace(0.5, 2.5, n_keys)
    n = int(3.0 * SAMPLE_RATE)
    audio = rng.normal(0, 0.001, size=n)
    t = np.arange(KEY_LENGTH) / SAMPLE_RATE
    click = np.sin(2 * np.pi * 2200 * t) * np.exp(-800 * t)
    for ts in timestamps:
        start = int(ts * SAMPLE_RATE)
        audio[start:start + KEY_LENGTH] += click
    return audio, timestamps


def test_extract_returns_exactly_46_features():
    audio, ts = _toy_signal()
    feats = extract_46_features(audio, ts, SAMPLE_RATE, KEY_LENGTH)
    assert feats.shape == (46,)
    assert np.all(np.isfinite(feats))


def test_extract_is_deterministic():
    audio, ts = _toy_signal(seed=42)
    a = extract_46_features(audio, ts, SAMPLE_RATE, KEY_LENGTH)
    b = extract_46_features(audio, ts, SAMPLE_RATE, KEY_LENGTH)
    np.testing.assert_array_equal(a, b)


def test_key_count_feature_matches_number_of_keys():
    audio, ts = _toy_signal(n_keys=10)
    feats = extract_46_features(audio, ts, SAMPLE_RATE, KEY_LENGTH)
    # index 44 is the key_count feature (32 MFCC + 6 strength + 6 timing = 44)
    assert feats[44] == pytest.approx(10.0)


def test_empty_audio_still_returns_46():
    feats = extract_46_features(np.zeros(0), np.zeros(0), SAMPLE_RATE, KEY_LENGTH)
    assert feats.shape == (46,)


def test_compute_mfcc_shape_and_finiteness():
    rng = np.random.default_rng(1)
    audio = rng.normal(0, 0.05, size=KEY_LENGTH)
    mfcc = compute_mfcc(audio, SAMPLE_RATE, num_coeffs=32, num_filters=32)
    assert mfcc.shape == (32,)
    assert np.all(np.isfinite(mfcc))


def test_compute_mfcc_empty_audio_returns_zeros():
    mfcc = compute_mfcc(np.zeros(0), SAMPLE_RATE, num_coeffs=32, num_filters=32)
    assert mfcc.shape == (32,)
    assert np.allclose(mfcc, 0.0)


def test_different_sounds_produce_different_mfcc():
    t = np.arange(KEY_LENGTH) / SAMPLE_RATE
    low = np.sin(2 * np.pi * 1000 * t)
    high = np.sin(2 * np.pi * 6000 * t)
    assert not np.allclose(
        compute_mfcc(low, SAMPLE_RATE), compute_mfcc(high, SAMPLE_RATE)
    )
