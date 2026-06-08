from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


MFCC_SLICE = slice(0, 32)
ACOUSTIC_SLICE = slice(32, 38)
TIMING_SLICE = slice(38, 44)
KEY_COUNT_INDEX = 44
TOTAL_TIME_INDEX = 45


@dataclass(slots=True)
class AttackSummary:
    name: str
    attempts: int
    accepted: int
    acceptance_rate: float
    min_score: float
    mean_score: float
    max_score: float
    acoustic_mean_score: float | None
    timing_mean_score: float | None
    max_balance: float | None
    threshold: float


@dataclass(slots=True)
class SimulationResult:
    config_path: str
    dataset_path: str
    model_path: str
    scaler_path: str
    threshold: float
    seed: int
    summaries: list[AttackSummary]


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed if seed is not None else 42)


def _clip_like_training(samples: np.ndarray, training: np.ndarray) -> np.ndarray:
    lower = np.percentile(training, 0.5, axis=0)
    upper = np.percentile(training, 99.5, axis=0)
    return np.clip(samples, lower, upper)


def _score_samples(artifacts, samples: np.ndarray, threshold: float):
    from keystroke_auth.modeling import evaluate_with_artifacts

    evaluations = [evaluate_with_artifacts(artifacts, row, threshold=threshold) for row in samples]
    joint_scores = np.array([item["scores"]["joint"] for item in evaluations], dtype=np.float64)
    accepted = int(sum(1 for item in evaluations if item["accepted"]))

    acoustic_scores = (
        np.array([item["scores"]["acoustic"] for item in evaluations], dtype=np.float64)
        if evaluations and "acoustic" in evaluations[0]["scores"]
        else None
    )
    timing_scores = (
        np.array([item["scores"]["timing"] for item in evaluations], dtype=np.float64)
        if evaluations and "timing" in evaluations[0]["scores"]
        else None
    )
    balance_scores = (
        np.array([item["scores"]["balance"] for item in evaluations], dtype=np.float64)
        if evaluations and "balance" in evaluations[0]["scores"]
        else None
    )
    return accepted, joint_scores, acoustic_scores, timing_scores, balance_scores


def hydra_equivalent_attack(training: np.ndarray, attempts: int, seed: int | None = None) -> np.ndarray:
    """Generate low-skill automated attempts with mechanical timing and weak acoustic data."""
    generator = _rng(seed)
    mean = np.mean(training, axis=0)
    std = np.std(training, axis=0)
    std = np.where(std <= 1e-9, 1e-6, std)

    samples = np.zeros((attempts, training.shape[1]), dtype=np.float64)
    samples[:, MFCC_SLICE] = generator.normal(0.0, 0.25, size=(attempts, 32))
    samples[:, ACOUSTIC_SLICE] = np.abs(generator.normal(0.0, 0.01, size=(attempts, 6)))

    timing_mean = max(float(mean[TIMING_SLICE.start]), 1e-4)
    samples[:, TIMING_SLICE] = generator.normal(
        timing_mean * 0.15,
        max(float(std[TIMING_SLICE.start]) * 0.05, 1e-4),
        size=(attempts, 6),
    )
    samples[:, KEY_COUNT_INDEX] = round(float(mean[KEY_COUNT_INDEX]))
    samples[:, TOTAL_TIME_INDEX] = generator.uniform(0.05, 0.35, size=attempts)
    return samples


def generative_timing_attack(
    training: np.ndarray, attempts: int, seed: int | None = None
) -> np.ndarray:
    """Generate attempts that mimic the owner's timing distribution but not acoustic texture."""
    generator = _rng(seed)
    mean = np.mean(training, axis=0)
    covariance = np.cov(training[:, TIMING_SLICE], rowvar=False)
    covariance = np.atleast_2d(covariance)
    covariance += np.eye(covariance.shape[0]) * 1e-6

    samples = np.tile(mean, (attempts, 1))
    random_rows = training[generator.integers(0, training.shape[0], size=attempts)]
    samples[:, ACOUSTIC_SLICE] = random_rows[:, ACOUSTIC_SLICE]
    samples[:, MFCC_SLICE] = generator.normal(
        np.mean(training[:, MFCC_SLICE], axis=0),
        np.std(training[:, MFCC_SLICE], axis=0) + 1e-6,
        size=(attempts, 32),
    )
    samples[:, TIMING_SLICE] = generator.multivariate_normal(
        mean[TIMING_SLICE], covariance, size=attempts
    )
    samples[:, KEY_COUNT_INDEX] = mean[KEY_COUNT_INDEX]
    samples[:, TOTAL_TIME_INDEX] = generator.normal(
        mean[TOTAL_TIME_INDEX],
        max(float(np.std(training[:, TOTAL_TIME_INDEX])), 1e-6),
        size=attempts,
    )
    return _clip_like_training(samples, training)


def acoustic_replay_attack(training: np.ndarray, attempts: int, seed: int | None = None) -> np.ndarray:
    """Replay acoustic features from owner rows while perturbing timing features."""
    generator = _rng(seed)
    source_rows = training[generator.integers(0, training.shape[0], size=attempts)]
    samples = source_rows.copy()

    timing_mean = np.mean(training[:, TIMING_SLICE], axis=0)
    timing_std = np.std(training[:, TIMING_SLICE], axis=0) + 1e-6
    samples[:, TIMING_SLICE] = generator.normal(
        timing_mean * generator.uniform(0.65, 1.45, size=(attempts, 1)),
        timing_std * generator.uniform(0.75, 2.0, size=(attempts, 1)),
    )
    samples[:, TOTAL_TIME_INDEX] = generator.normal(
        np.mean(training[:, TOTAL_TIME_INDEX]) * generator.uniform(0.65, 1.45, size=attempts),
        np.std(training[:, TOTAL_TIME_INDEX]) + 1e-6,
        size=attempts,
    )
    return _clip_like_training(samples, training)


ATTACK_GENERATORS = {
    "hydra_equivalent": hydra_equivalent_attack,
    "generative_timing": generative_timing_attack,
    "acoustic_replay": acoustic_replay_attack,
}


def run_simulation(
    dataset_path: str | Path = "dataset.csv",
    model_path: str | Path = "model.pkl",
    scaler_path: str | Path = "scaler.pkl",
    config_path: str | Path = "config.json",
    attempts: int = 250,
    threshold: float | None = None,
    seed: int = 42,
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
        samples = ATTACK_GENERATORS[name](training, attempts, seed + index)
        accepted, scores, acoustic_scores, timing_scores, balance_scores = _score_samples(
            artifacts, samples, threshold_value
        )
        summaries.append(
            AttackSummary(
                name=name,
                attempts=attempts,
                accepted=accepted,
                acceptance_rate=accepted / attempts if attempts else 0.0,
                min_score=float(np.min(scores)),
                mean_score=float(np.mean(scores)),
                max_score=float(np.max(scores)),
                acoustic_mean_score=(
                    float(np.mean(acoustic_scores)) if acoustic_scores is not None else None
                ),
                timing_mean_score=(
                    float(np.mean(timing_scores)) if timing_scores is not None else None
                ),
                max_balance=(
                    float(np.max(balance_scores)) if balance_scores is not None else None
                ),
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
                    "acoustic_mean_score",
                    "timing_mean_score",
                    "max_balance",
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
                    "Subscores: "
                    f"acoustic_mean={summary.acoustic_mean_score:.6f}, "
                    f"timing_mean={summary.timing_mean_score:.6f}, "
                    f"max_balance={summary.max_balance:.6f}"
                )
                if summary.acoustic_mean_score is not None
                and summary.timing_mean_score is not None
                and summary.max_balance is not None
                else "Subscores: unavailable",
                "",
            ]
        )
    return "\n".join(lines).rstrip()
