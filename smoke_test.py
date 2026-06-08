"""
Headless smoke test for the keystroke + acoustic authentication pipeline.

This script needs ONLY numpy + scikit-learn (no microphone, no GUI, no pyaudio,
no pynput). It synthesizes an "owner" typing/sound profile and an "imposter"
profile, runs the real feature-extraction and modeling code, then reports
FAR / FRR / accuracy for every supported algorithm.

Run:
    python smoke_test.py
"""

from __future__ import annotations

import numpy as np

from keystroke_auth.features import extract_46_features
from keystroke_auth.modeling import (
    calibrate_auth_threshold,
    score_with_artifacts,
    train_one_class_model,
)

SAMPLE_RATE = 44100
KEY_LENGTH = 4500
N_KEYS = 12            # length of the "passphrase"
DURATION_SEC = 4.0


def _synth_sample(rng: np.ndarray, *, base_interval: float, jitter: float,
                  click_freq: float, decay: float) -> np.ndarray:
    """Build one (audio, timestamps) sample that mimics a person typing.

    A person is characterized by: typing rhythm (base_interval/jitter) and the
    spectral signature of their key clicks (click_freq/decay). Owner and
    imposter differ in BOTH timing and sound -- exactly the two modalities the
    real system fuses.
    """
    # --- timing (keystroke dynamics) ---
    intervals = base_interval + rng.normal(0.0, jitter, size=N_KEYS - 1)
    intervals = np.clip(intervals, 0.03, None)
    timestamps = np.concatenate([[0.5], 0.5 + np.cumsum(intervals)])

    # --- audio (acoustic signal) ---
    n = int(DURATION_SEC * SAMPLE_RATE)
    audio = rng.normal(0.0, 0.001, size=n)  # ambient noise floor
    t = np.arange(KEY_LENGTH) / SAMPLE_RATE
    click = np.sin(2 * np.pi * click_freq * t) * np.exp(-decay * t)
    for ts in timestamps:
        start = int(ts * SAMPLE_RATE)
        stop = min(start + KEY_LENGTH, n)
        audio[start:stop] += click[: stop - start]
    return audio, timestamps


def make_dataset(n_samples: int, profile: dict, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_samples):
        audio, timestamps = _synth_sample(
            rng,
            base_interval=profile["base_interval"],
            jitter=profile["jitter"],
            click_freq=profile["click_freq"],
            decay=profile["decay"],
        )
        rows.append(
            extract_46_features(audio, timestamps, SAMPLE_RATE, KEY_LENGTH)
        )
    return np.vstack(rows)


OWNER = dict(base_interval=0.18, jitter=0.015, click_freq=2200.0, decay=900.0)
IMPOSTER = dict(base_interval=0.12, jitter=0.040, click_freq=3500.0, decay=500.0)


def main() -> None:
    print("Synthesizing data (this runs the REAL 46-D feature extractor)...")
    train = make_dataset(40, OWNER, seed=1)
    owner_test = make_dataset(20, OWNER, seed=2)
    imposter_test = make_dataset(20, IMPOSTER, seed=3)
    print(f"  feature matrix shape: {train.shape}  (expect (40, 46))\n")

    header = f"{'algorithm':>10} | {'threshold':>10} | {'FAR':>7} | {'FRR':>7} | {'accuracy':>8}"
    print(header)
    print("-" * len(header))

    for algo in ("oc_svm", "pca_svm", "lof", "iforest"):
        artifacts = train_one_class_model(train, algorithm=algo, nu=0.05, gamma=0.001)
        owner_scores = [score_with_artifacts(artifacts, r) for r in owner_test]
        imposter_scores = [score_with_artifacts(artifacts, r) for r in imposter_test]
        thr, far, frr = calibrate_auth_threshold(owner_scores, imposter_scores)

        correct = sum(s > thr for s in owner_scores) + sum(s <= thr for s in imposter_scores)
        acc = correct / (len(owner_scores) + len(imposter_scores))
        print(f"{algo:>10} | {thr:>10.4f} | {far*100:>6.1f}% | {frr*100:>6.1f}% | {acc*100:>7.1f}%")

    print("\nOK: extraction + training + scoring + FAR/FRR pipeline all ran end-to-end.")


if __name__ == "__main__":
    main()
