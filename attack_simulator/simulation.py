from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

MFCC_SLICE = slice(0, 32)
ENERGY_SLICE = slice(32, 38)
ACOUSTIC_SLICE = slice(0, 38)
TIMING_SLICE = slice(38, 44)
KEY_COUNT_INDEX = 44
TOTAL_TIME_INDEX = 45

REALISM_PROFILES = {
    "casual": {
        "timing_mix": 0.3,
        "timing_cov_scale": 2.0,
        "timing_shift": 0.2,
        "acoustic_mix": 0.2,
        "acoustic_cov_scale": 2.3,
        "channel_noise": 0.22,
        "energy_reuse_prob": 0.1,
        "key_count_noise": 1.6,
        "total_time_scale": 1.8,
    },
    "practical": {
        "timing_mix": 0.48,
        "timing_cov_scale": 1.45,
        "timing_shift": 0.1,
        "acoustic_mix": 0.4,
        "acoustic_cov_scale": 1.6,
        "channel_noise": 0.12,
        "energy_reuse_prob": 0.32,
        "key_count_noise": 0.85,
        "total_time_scale": 1.3,
    },
    "idealized": {
        "timing_mix": 0.72,
        "timing_cov_scale": 1.0,
        "timing_shift": 0.03,
        "acoustic_mix": 0.68,
        "acoustic_cov_scale": 1.0,
        "channel_noise": 0.04,
        "energy_reuse_prob": 0.78,
        "key_count_noise": 0.25,
        "total_time_scale": 1.05,
    },
}


@dataclass(slots=True)
class AttackSummary:
    name: str
    attempts: int
    accepted: int
    acceptance_rate: float
    min_score: float
    mean_score: float
    max_score: float
    acoustic_distance_mean: float
    timing_distance_mean: float
    threshold: float


@dataclass(slots=True)
class SimulationResult:
    config_path: str
    dataset_path: str
    model_path: str
    scaler_path: str
    threshold: float
    seed: int
    realism: str
    summaries: list[AttackSummary]


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed if seed is not None else 42)


def _profile(name: str) -> dict[str, float]:
    if name not in REALISM_PROFILES:
        raise ValueError(f"Unknown realism profile: {name}")
    return REALISM_PROFILES[name]


def _clip_like_training(samples: np.ndarray, training: np.ndarray) -> np.ndarray:
    lower = np.percentile(training, 0.5, axis=0)
    upper = np.percentile(training, 99.5, axis=0)
    return np.clip(samples, lower, upper)


def _score_samples(artifacts, samples: np.ndarray, threshold: float) -> tuple[int, np.ndarray]:
    from keystroke_auth.modeling import score_with_artifacts

    joint_scores = np.array([score_with_artifacts(artifacts, row) for row in samples], dtype=np.float64)
    accepted = int(np.sum(joint_scores > threshold))
    return accepted, joint_scores


def _mean_normalized_distance(
    samples: np.ndarray, training: np.ndarray, feature_slice: slice
) -> float:
    reference = training[:, feature_slice]
    if reference.size == 0:
        return 0.0
    mean = np.mean(reference, axis=0)
    std = np.std(reference, axis=0)
    std = np.where(std <= 1e-9, 1.0, std)
    zscores = (samples[:, feature_slice] - mean) / std
    return float(np.mean(np.linalg.norm(zscores, axis=1)))


def _apply_channel_mismatch(
    acoustic_features: np.ndarray,
    generator: np.random.Generator,
    severity: float,
) -> np.ndarray:
    transformed = acoustic_features.copy()
    mfcc = transformed[:, MFCC_SLICE]
    energies = transformed[:, ENERGY_SLICE]

    spectral_tilt = np.linspace(
        1.0 + severity * 0.12,
        max(0.55, 1.0 - severity * 0.18),
        mfcc.shape[1],
    )
    mfcc *= spectral_tilt
    mfcc += generator.normal(0.0, severity * 0.22, size=mfcc.shape)
    mfcc *= generator.normal(1.0, severity * 0.16, size=(mfcc.shape[0], 1))

    energies *= generator.normal(1.0, severity * 0.18, size=(energies.shape[0], 1))
    energies += generator.normal(0.0, severity * 0.05, size=energies.shape)
    energies[:] = np.clip(energies, 0.0, None)
    return transformed


def hydra_equivalent_attack(
    training: np.ndarray,
    attempts: int,
    seed: int | None = None,
    realism: str = "practical",
) -> np.ndarray:
    generator = _rng(seed)
    profile = _profile(realism)
    mean = np.mean(training, axis=0)
    std = np.std(training, axis=0)
    std = np.where(std <= 1e-9, 1e-6, std)

    samples = np.zeros((attempts, training.shape[1]), dtype=np.float64)
    samples[:, MFCC_SLICE] = generator.normal(
        0.0, 0.3 + profile["channel_noise"], size=(attempts, MFCC_SLICE.stop - MFCC_SLICE.start)
    )
    samples[:, ENERGY_SLICE] = np.abs(
        generator.normal(
            0.0,
            0.02 + profile["channel_noise"] * 0.04,
            size=(attempts, ENERGY_SLICE.stop - ENERGY_SLICE.start),
        )
    )
    timing_mean = max(float(mean[TIMING_SLICE.start]), 1e-4)
    samples[:, TIMING_SLICE] = generator.normal(
        timing_mean * (0.15 + profile["timing_shift"]),
        max(float(std[TIMING_SLICE.start]) * (0.08 + profile["timing_shift"]), 1e-4),
        size=(attempts, TIMING_SLICE.stop - TIMING_SLICE.start),
    )
    samples[:, KEY_COUNT_INDEX] = np.round(
        generator.normal(float(mean[KEY_COUNT_INDEX]), max(profile["key_count_noise"], 0.5), size=attempts)
    )
    samples[:, TOTAL_TIME_INDEX] = generator.uniform(
        0.05, 0.35 * profile["total_time_scale"], size=attempts
    )
    return samples


def generative_timing_attack(
    training: np.ndarray,
    attempts: int,
    seed: int | None = None,
    realism: str = "practical",
) -> np.ndarray:
    generator = _rng(seed)
    profile = _profile(realism)
    mean = np.mean(training, axis=0)

    timing_cov = np.cov(training[:, TIMING_SLICE], rowvar=False)
    timing_cov = np.atleast_2d(timing_cov)
    timing_cov += np.eye(timing_cov.shape[0]) * 1e-6

    acoustic_cov = np.cov(training[:, ACOUSTIC_SLICE], rowvar=False)
    acoustic_cov = np.atleast_2d(acoustic_cov)
    acoustic_cov += np.eye(acoustic_cov.shape[0]) * 1e-5

    samples = np.tile(mean, (attempts, 1))
    acoustic_rows = training[generator.integers(0, training.shape[0], size=attempts), ACOUSTIC_SLICE]
    acoustic_draw = generator.multivariate_normal(
        np.mean(training[:, ACOUSTIC_SLICE], axis=0),
        acoustic_cov * profile["acoustic_cov_scale"],
        size=attempts,
    )
    samples[:, ACOUSTIC_SLICE] = (
        profile["acoustic_mix"] * acoustic_rows + (1.0 - profile["acoustic_mix"]) * acoustic_draw
    )
    keep_mask = generator.random(attempts) < profile["energy_reuse_prob"]
    if np.any(keep_mask):
        samples[keep_mask, ENERGY_SLICE] = acoustic_rows[keep_mask, ENERGY_SLICE]
    samples[:, ACOUSTIC_SLICE] = _apply_channel_mismatch(
        samples[:, ACOUSTIC_SLICE], generator, profile["channel_noise"]
    )

    timing_rows = training[generator.integers(0, training.shape[0], size=attempts), TIMING_SLICE]
    timing_draw = generator.multivariate_normal(
        np.mean(training[:, TIMING_SLICE], axis=0) * (1.0 + profile["timing_shift"]),
        timing_cov * profile["timing_cov_scale"],
        size=attempts,
    )
    samples[:, TIMING_SLICE] = (
        profile["timing_mix"] * timing_rows + (1.0 - profile["timing_mix"]) * timing_draw
    )
    samples[:, KEY_COUNT_INDEX] = np.round(
        generator.normal(
            float(mean[KEY_COUNT_INDEX]),
            max(float(np.std(training[:, KEY_COUNT_INDEX])) * profile["key_count_noise"], 0.35),
            size=attempts,
        )
    )
    samples[:, TOTAL_TIME_INDEX] = generator.normal(
        float(mean[TOTAL_TIME_INDEX]),
        max(float(np.std(training[:, TOTAL_TIME_INDEX])) * profile["total_time_scale"], 1e-6),
        size=attempts,
    )
    return _clip_like_training(samples, training)


def acoustic_replay_attack(
    training: np.ndarray,
    attempts: int,
    seed: int | None = None,
    realism: str = "practical",
) -> np.ndarray:
    generator = _rng(seed)
    profile = _profile(realism)
    source_rows = training[generator.integers(0, training.shape[0], size=attempts)]
    samples = source_rows.copy()
    samples[:, ACOUSTIC_SLICE] = _apply_channel_mismatch(
        samples[:, ACOUSTIC_SLICE], generator, profile["channel_noise"]
    )

    timing_mean = np.mean(training[:, TIMING_SLICE], axis=0)
    timing_std = np.std(training[:, TIMING_SLICE], axis=0) + 1e-6
    samples[:, TIMING_SLICE] = generator.normal(
        timing_mean
        * generator.uniform(
            0.65 - profile["timing_shift"], 1.45 + profile["timing_shift"], size=(attempts, 1)
        ),
        timing_std
        * generator.uniform(0.9, 2.1 * profile["timing_cov_scale"], size=(attempts, 1)),
    )
    samples[:, KEY_COUNT_INDEX] = np.round(
        generator.normal(
            float(np.mean(training[:, KEY_COUNT_INDEX])),
            max(float(np.std(training[:, KEY_COUNT_INDEX])) * profile["key_count_noise"], 0.4),
            size=attempts,
        )
    )
    samples[:, TOTAL_TIME_INDEX] = generator.normal(
        float(np.mean(training[:, TOTAL_TIME_INDEX]))
        * generator.uniform(
            0.65 - profile["timing_shift"], 1.45 + profile["timing_shift"], size=attempts
        ),
        float(np.std(training[:, TOTAL_TIME_INDEX])) * profile["total_time_scale"] + 1e-6,
        size=attempts,
    )
    return _clip_like_training(samples, training)


def synthetic_acoustic_attack(
    training: np.ndarray,
    attempts: int,
    seed: int | None = None,
    realism: str = "practical",
) -> np.ndarray:
    generator = _rng(seed)
    profile = _profile(realism)
    samples = np.zeros((attempts, training.shape[1]), dtype=np.float64)

    acoustic_reference = training[:, ACOUSTIC_SLICE]
    acoustic_mean = np.mean(acoustic_reference, axis=0)
    acoustic_covariance = np.cov(acoustic_reference, rowvar=False)
    acoustic_covariance = np.atleast_2d(acoustic_covariance)
    acoustic_covariance += np.eye(acoustic_covariance.shape[0]) * 1e-5

    source_a = training[generator.integers(0, training.shape[0], size=attempts), ACOUSTIC_SLICE]
    source_b = training[generator.integers(0, training.shape[0], size=attempts), ACOUSTIC_SLICE]
    mix = generator.uniform(0.2, 0.8, size=(attempts, 1))
    prototype_blend = (mix * source_a) + ((1.0 - mix) * source_b)
    gaussian_component = generator.multivariate_normal(
        acoustic_mean,
        acoustic_covariance * profile["acoustic_cov_scale"],
        size=attempts,
    )
    samples[:, ACOUSTIC_SLICE] = (
        profile["acoustic_mix"] * prototype_blend
        + (1.0 - profile["acoustic_mix"]) * gaussian_component
    )
    samples[:, ACOUSTIC_SLICE] = _apply_channel_mismatch(
        samples[:, ACOUSTIC_SLICE], generator, profile["channel_noise"]
    )

    timing_reference = training[:, TIMING_SLICE]
    timing_mean = np.mean(timing_reference, axis=0)
    timing_covariance = np.cov(timing_reference, rowvar=False)
    timing_covariance = np.atleast_2d(timing_covariance)
    timing_covariance += np.eye(timing_covariance.shape[0]) * 1e-6
    samples[:, TIMING_SLICE] = generator.multivariate_normal(
        timing_mean * (1.0 + profile["timing_shift"]),
        timing_covariance * max(1.35, profile["timing_cov_scale"]),
        size=attempts,
    )

    key_count_mean = float(np.mean(training[:, KEY_COUNT_INDEX]))
    key_count_std = max(float(np.std(training[:, KEY_COUNT_INDEX])), 1e-6)
    samples[:, KEY_COUNT_INDEX] = np.round(
        generator.normal(
            key_count_mean,
            max(key_count_std * profile["key_count_noise"], 0.35),
            size=attempts,
        )
    )

    total_time_mean = float(np.mean(training[:, TOTAL_TIME_INDEX]))
    total_time_std = max(float(np.std(training[:, TOTAL_TIME_INDEX])), 1e-6)
    samples[:, TOTAL_TIME_INDEX] = generator.normal(
        total_time_mean,
        total_time_std * max(1.25, profile["total_time_scale"]),
        size=attempts,
    )
    return _clip_like_training(samples, training)


ATTACK_GENERATORS = {
    "hydra_equivalent": hydra_equivalent_attack,
    "generative_timing": generative_timing_attack,
    "acoustic_replay": acoustic_replay_attack,
    "synthetic_acoustic": synthetic_acoustic_attack,
}


def run_simulation(
    dataset_path: str | Path = "dataset.csv",
    model_path: str | Path = "model.pkl",
    scaler_path: str | Path = "scaler.pkl",
    config_path: str | Path = "config.json",
    attempts: int = 250,
    threshold: float | None = None,
    seed: int = 42,
    realism: str = "practical",
    attack_names: Iterable[str] | None = None,
) -> SimulationResult:
    from keystroke_auth.config import load_config
    from keystroke_auth.modeling import load_artifacts, load_feature_matrix

    dataset_path = Path(dataset_path)
    model_path = Path(model_path)
    scaler_path = Path(scaler_path)
    config_path = Path(config_path)

    config = load_config(config_path)
    threshold_value = float(config.auth_threshold if threshold is None else threshold)
    training = load_feature_matrix(dataset_path)
    if training.shape[0] == 0:
        raise ValueError(f"Training dataset is empty: {dataset_path}")

    artifacts = load_artifacts(model_path, scaler_path)
    selected_attacks = list(attack_names or ATTACK_GENERATORS.keys())
    summaries: list[AttackSummary] = []

    for index, name in enumerate(selected_attacks):
        if name not in ATTACK_GENERATORS:
            raise ValueError(f"Unknown attack simulation: {name}")
        samples = ATTACK_GENERATORS[name](training, attempts, seed + index, realism=realism)
        accepted, scores = _score_samples(artifacts, samples, threshold_value)
        summaries.append(
            AttackSummary(
                name=name,
                attempts=attempts,
                accepted=accepted,
                acceptance_rate=accepted / attempts if attempts else 0.0,
                min_score=float(np.min(scores)),
                mean_score=float(np.mean(scores)),
                max_score=float(np.max(scores)),
                acoustic_distance_mean=_mean_normalized_distance(samples, training, ACOUSTIC_SLICE),
                timing_distance_mean=_mean_normalized_distance(samples, training, TIMING_SLICE),
                threshold=threshold_value,
            )
        )

    return SimulationResult(
        config_path=str(config_path),
        dataset_path=str(dataset_path),
        model_path=str(model_path),
        scaler_path=str(scaler_path),
        threshold=threshold_value,
        seed=seed,
        realism=realism,
        summaries=summaries,
    )


def save_result(result: SimulationResult, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "name",
                    "attempts",
                    "accepted",
                    "acceptance_rate",
                    "min_score",
                    "mean_score",
                    "max_score",
                    "acoustic_distance_mean",
                    "timing_distance_mean",
                    "threshold",
                ],
            )
            writer.writeheader()
            for summary in result.summaries:
                writer.writerow(asdict(summary))
        return

    payload = asdict(result)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def format_result(result: SimulationResult) -> str:
    lines = [
        "Attack Simulation & Validation",
        f"Dataset: {result.dataset_path}",
        f"Model: {result.model_path}",
        f"Scaler: {result.scaler_path}",
        f"Threshold: {result.threshold:.6f}",
        f"Realism: {result.realism}",
        "",
    ]
    for summary in result.summaries:
        lines.extend(
            [
                f"[{summary.name}]",
                f"Attempts: {summary.attempts}",
                f"Accepted: {summary.accepted}",
                f"Acceptance Rate: {summary.acceptance_rate * 100:.2f}%",
                (
                    "Scores: "
                    f"min={summary.min_score:.6f}, "
                    f"mean={summary.mean_score:.6f}, "
                    f"max={summary.max_score:.6f}"
                ),
                (
                    "Feature Distance: "
                    f"acoustic_mean={summary.acoustic_distance_mean:.6f}, "
                    f"timing_mean={summary.timing_distance_mean:.6f}"
                ),
                "",
            ]
        )
    return "\n".join(lines).rstrip()
