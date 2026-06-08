"""Synthetic data generator for offline experiments.

Produces realistic 46-D feature rows by running the REAL feature extractor on
synthesized audio + keystroke timings. Owner and imposter differ in BOTH typing
rhythm and key-click acoustics, but only *moderately*, so the resulting ROC/AUC
is realistic (not a trivial 100%).

This lets the experiment scripts run end-to-end with zero hardware. When you
later collect real samples from classmates, pass the CSVs instead and the same
analysis runs on real data.
"""

from __future__ import annotations

import numpy as np

from keystroke_auth.features import extract_46_features

SAMPLE_RATE = 44100
KEY_LENGTH = 4500
N_KEYS = 12
DURATION_SEC = 4.0

# Each "person" is a profile: typing rhythm + key-click sound signature.
OWNER = dict(base_interval=0.165, jitter=0.028, click_freq=2300.0, decay=820.0)
# Imposter is deliberately *similar* (knows the passphrase, types at a humanlike
# pace, on a similar keyboard) so neither modality alone is perfectly separable
# and their FUSION is what wins. This keeps the ROC/AUC realistic.
IMPOSTER = dict(base_interval=0.150, jitter=0.040, click_freq=2355.0, decay=775.0)


def _synth_sample(rng: np.random.Generator, *, base_interval: float, jitter: float,
                  click_freq: float, decay: float) -> tuple[np.ndarray, np.ndarray]:
    intervals = np.clip(base_interval + rng.normal(0.0, jitter, size=N_KEYS - 1), 0.03, None)
    timestamps = np.concatenate([[0.5], 0.5 + np.cumsum(intervals)])

    n = int(DURATION_SEC * SAMPLE_RATE)
    audio = rng.normal(0.0, 0.001, size=n)
    t = np.arange(KEY_LENGTH) / SAMPLE_RATE
    # small per-sample variation in the click so samples are not identical
    click = np.sin(2 * np.pi * click_freq * (1 + rng.normal(0, 0.035)) * t) * np.exp(-decay * t)
    for ts in timestamps:
        start = int(ts * SAMPLE_RATE)
        stop = min(start + KEY_LENGTH, n)
        audio[start:stop] += click[: stop - start]
    return audio, timestamps


def make_dataset(n_samples: int, profile: dict, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = [
        extract_46_features(*_synth_sample(rng, **profile), SAMPLE_RATE, KEY_LENGTH)
        for _ in range(n_samples)
    ]
    return np.vstack(rows)


def make_owner(n: int, seed: int) -> np.ndarray:
    return make_dataset(n, OWNER, seed)


def make_imposter(n: int, seed: int) -> np.ndarray:
    return make_dataset(n, IMPOSTER, seed)
