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
        "timing_jitter": 0.58,
        "timing_shift": 0.18,
        "acoustic_noise": 0.30,
        "prototype_mix": 0.24,
        "energy_noise": 0.22,
        "tempo_control": 0.28,
        "hydra_humanization": 0.20,
        "hydra_speed": 0.48,
    },
    "practical": {
        "timing_jitter": 0.36,
        "timing_shift": 0.09,
        "acoustic_noise": 0.18,
        "prototype_mix": 0.40,
        "energy_noise": 0.13,
        "tempo_control": 0.48,
        "hydra_humanization": 0.35,
        "hydra_speed": 0.62,
    },
}

# Broad public priors for keyboard-click MFCC-like shapes. They are intentionally
# generic and are not estimated from the owner dataset or model internals.
KEYBOARD_PROTOTYPES = np.array(
    [
        [
            2.4, 1.8, 1.1, 0.6, 0.2, -0.2, -0.6, -0.9,
            -1.1, -1.2, -1.1, -1.0, -0.9, -0.8, -0.8, -0.7,
            -0.7, -0.6, -0.5, -0.4, -0.4, -0.3, -0.3, -0.2,
            -0.2, -0.1, -0.1, 0.0, 0.0, 0.1, 0.1, 0.1,
        ],
        [
            1.7, 1.4, 1.2, 0.9, 0.5, 0.1, -0.2, -0.4,
            -0.6, -0.8, -0.9, -0.8, -0.7, -0.5, -0.4, -0.3,
            -0.2, -0.1, -0.1, 0.0, 0.0, 0.1, 0.1, 0.1,
            0.2, 0.2, 0.2, 0.1, 0.1, 0.0, -0.1, -0.1,
        ],
        [
            2.0, 1.5, 0.8, 0.2, -0.1, -0.3, -0.5, -0.7,
            -0.8, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.3,
            -0.2, -0.2, -0.1, -0.1, -0.1, 0.0, 0.0, 0.1,
            0.1, 0.0, 0.0, -0.1, -0.1, -0.2, -0.2, -0.3,
        ],
    ],
    dtype=np.float64,
)


@dataclass(slots=True)
class AttackContext:
    key_count_hint: float
    total_time_hint: float


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


def _context_from_passphrase(passphrase: str) -> AttackContext:
    key_count = float(max(len(passphrase.strip()), 1))
    # Public typing-speed prior, not owner-derived: roughly 4.5 chars/sec with slack.
    total_time = max(key_count / 4.5, 0.45)
    return AttackContext(key_count_hint=key_count, total_time_hint=total_time)


def _clip_public_bounds(samples: np.ndarray, context: AttackContext) -> np.ndarray:
    bounded = samples.copy()
    bounded[:, MFCC_SLICE] = np.clip(bounded[:, MFCC_SLICE], -8.0, 8.0)
    bounded[:, ENERGY_SLICE] = np.clip(bounded[:, ENERGY_SLICE], 0.0, 4.0)
    bounded[:, TIMING_SLICE] = np.clip(bounded[:, TIMING_SLICE], 0.0, 1.0)
    bounded[:, KEY_COUNT_INDEX] = np.clip(
        np.round(bounded[:, KEY_COUNT_INDEX]),
        max(context.key_count_hint - 2.0, 1.0),
        context.key_count_hint + 2.0,
    )
    bounded[:, TOTAL_TIME_INDEX] = np.clip(
        bounded[:, TOTAL_TIME_INDEX],
        max(context.total_time_hint * 0.35, 0.15),
        context.total_time_hint * 2.2,
    )
    return bounded


def _score_samples(artifacts, samples: np.ndarray, threshold: float) -> tuple[int, np.ndarray]:
    from keystroke_auth.modeling import score_with_artifacts

    scores = np.array([score_with_artifacts(artifacts, row) for row in samples], dtype=np.float64)
    accepted = int(np.sum(scores > threshold))
    return accepted, scores


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


def _generic_keyboard_acoustics(
    attempts: int,
    generator: np.random.Generator,
    realism: str,
) -> np.ndarray:
    profile = _profile(realism)
    prototype_ids = generator.integers(0, len(KEYBOARD_PROTOTYPES), size=attempts)
    prototypes = KEYBOARD_PROTOTYPES[prototype_ids]
    secondary = KEYBOARD_PROTOTYPES[generator.integers(0, len(KEYBOARD_PROTOTYPES), size=attempts)]
    mix = generator.uniform(0.0, profile["prototype_mix"], size=(attempts, 1))
    mfcc = ((1.0 - mix) * prototypes) + (mix * secondary)

    tilt = np.linspace(0.18, -0.12, MFCC_SLICE.stop - MFCC_SLICE.start)
    mfcc += tilt
    mfcc += generator.normal(0.0, profile["acoustic_noise"], size=mfcc.shape)

    energies = np.abs(
        generator.normal(
            loc=0.26,
            scale=profile["energy_noise"],
            size=(attempts, ENERGY_SLICE.stop - ENERGY_SLICE.start),
        )
    )
    energy_shape = generator.uniform(0.75, 1.35, size=(attempts, 1))
    energies *= energy_shape
    return np.concatenate([mfcc, energies], axis=1)


def _feature_aware_timing(
    attempts: int,
    generator: np.random.Generator,
    realism: str,
    context: AttackContext,
    style: str,
) -> tuple[np.ndarray, np.ndarray]:
    profile = _profile(realism)
    if style == "scripted":
        base = np.array([0.055, 0.018, 0.12, 0.01, 0.045, 0.045], dtype=np.float64)
        spread = np.array([0.018, 0.012, 0.026, 0.008, 0.018, 0.018], dtype=np.float64)
        total_center = context.total_time_hint * 0.58
    elif style == "practiced":
        base = np.array([0.13, 0.08, 0.27, 0.035, 0.11, 0.095], dtype=np.float64)
        spread = np.array([0.055, 0.04, 0.085, 0.03, 0.045, 0.045], dtype=np.float64)
        total_center = context.total_time_hint * 1.05
    else:
        base = np.array([0.10, 0.06, 0.21, 0.025, 0.085, 0.08], dtype=np.float64)
        spread = np.array([0.045, 0.035, 0.07, 0.025, 0.04, 0.04], dtype=np.float64)
        total_center = context.total_time_hint * 1.12

    shift = np.array([0.08, 0.0, 0.12, -0.025, 0.02, 0.01], dtype=np.float64)
    loc = base + (shift * profile["timing_shift"])
    scale = spread * (1.0 + profile["timing_jitter"])
    timing = generator.normal(loc=loc, scale=scale, size=(attempts, TIMING_SLICE.stop - TIMING_SLICE.start))
    timing = np.clip(timing, 0.0, None)

    tempo_control = profile["tempo_control"]
    total_time = generator.normal(
        loc=total_center,
        scale=max(context.total_time_hint * (0.28 - tempo_control * 0.12), 0.05),
        size=attempts,
    )
    total_time = np.clip(total_time, 0.12, None)
    return timing, total_time


def _hydra_base_attempts(
    context: AttackContext,
    attempts: int,
    generator: np.random.Generator,
    realism: str = "practical",
) -> np.ndarray:
    profile = _profile(realism)
    acoustic = _generic_keyboard_acoustics(attempts, generator, realism)

    base = np.array([0.045, 0.012, 0.095, 0.008, 0.035, 0.035], dtype=np.float64)
    spread = np.array([0.012, 0.008, 0.018, 0.006, 0.012, 0.012], dtype=np.float64)
    timing = generator.normal(
        loc=base,
        scale=spread * (1.0 + profile["timing_jitter"] * 0.35),
        size=(attempts, TIMING_SLICE.stop - TIMING_SLICE.start),
    )
    timing = np.clip(timing, 0.0, None)

    total_time = generator.normal(
        loc=max(context.total_time_hint * profile["hydra_speed"], 0.18),
        scale=max(context.total_time_hint * 0.06, 0.025),
        size=attempts,
    )
    key_count = generator.normal(context.key_count_hint, 0.18, size=attempts)

    samples = np.zeros((attempts, 46), dtype=np.float64)
    samples[:, ACOUSTIC_SLICE] = acoustic
    samples[:, TIMING_SLICE] = timing
    samples[:, KEY_COUNT_INDEX] = key_count
    samples[:, TOTAL_TIME_INDEX] = total_time
    return _clip_public_bounds(samples, context)


def hydra_scripted_burst_attack(
    context: AttackContext,
    attempts: int,
    seed: int | None = None,
    realism: str = "practical",
) -> np.ndarray:
    """Hydra-style burst: fast automated attempts with generic keyboard acoustics."""
    generator = _rng(seed)
    samples = _hydra_base_attempts(context, attempts, generator, realism)
    samples[:, TIMING_SLICE] *= generator.uniform(0.85, 1.18, size=(attempts, 1))
    samples[:, ENERGY_SLICE] *= generator.uniform(0.75, 1.25, size=(attempts, 1))
    return _clip_public_bounds(samples, context)


def hydra_humanized_timing_attack(
    context: AttackContext,
    attempts: int,
    seed: int | None = None,
    realism: str = "practical",
) -> np.ndarray:
    """Hydra-style automation with coarse human-like pauses added between attempts."""
    generator = _rng(seed)
    profile = _profile(realism)
    samples = _hydra_base_attempts(context, attempts, generator, realism)
    human_timing, human_total_time = _feature_aware_timing(
        attempts, generator, realism, context, "practiced"
    )
    blend = profile["hydra_humanization"]
    samples[:, TIMING_SLICE] = ((1.0 - blend) * samples[:, TIMING_SLICE]) + (blend * human_timing)
    samples[:, TOTAL_TIME_INDEX] = (
        (1.0 - blend) * samples[:, TOTAL_TIME_INDEX] + blend * human_total_time
    )
    samples[:, ACOUSTIC_SLICE] += generator.normal(
        0.0, profile["acoustic_noise"] * 0.25, size=(attempts, ACOUSTIC_SLICE.stop)
    )
    return _clip_public_bounds(samples, context)


def hydra_synthetic_keyboard_attack(
    context: AttackContext,
    attempts: int,
    seed: int | None = None,
    realism: str = "practical",
) -> np.ndarray:
    """Hydra-style automation with generic synthetic keyboard-click shaping."""
    generator = _rng(seed)
    profile = _profile(realism)
    samples = _hydra_base_attempts(context, attempts, generator, realism)
    samples[:, MFCC_SLICE] += np.linspace(0.42, -0.30, MFCC_SLICE.stop - MFCC_SLICE.start)
    samples[:, MFCC_SLICE] += generator.normal(
        0.0, profile["acoustic_noise"] * 0.35, size=(attempts, MFCC_SLICE.stop)
    )
    samples[:, ENERGY_SLICE] *= generator.uniform(0.85, 1.55, size=(attempts, 1))
    synthetic_timing, synthetic_total_time = _feature_aware_timing(
        attempts, generator, realism, context, "synthetic"
    )
    blend = profile["hydra_humanization"] * 0.75
    samples[:, TIMING_SLICE] = (
        (1.0 - blend) * samples[:, TIMING_SLICE] + blend * synthetic_timing
    )
    samples[:, TOTAL_TIME_INDEX] = (
        (1.0 - blend) * samples[:, TOTAL_TIME_INDEX] + blend * synthetic_total_time
    )
    return _clip_public_bounds(samples, context)


ATTACK_GENERATORS = {
    "hydra_scripted_burst": hydra_scripted_burst_attack,
    "hydra_humanized_timing": hydra_humanized_timing_attack,
    "hydra_synthetic_keyboard": hydra_synthetic_keyboard_attack,
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
    context = _context_from_passphrase(config.passphrase)
    training = load_feature_matrix(dataset_path)
    if training.shape[0] == 0:
        raise ValueError(f"Training dataset is empty: {dataset_path}")

    artifacts = load_artifacts(model_path, scaler_path)
    selected_attacks = list(attack_names or ATTACK_GENERATORS.keys())
    summaries: list[AttackSummary] = []

    for index, name in enumerate(selected_attacks):
        if name not in ATTACK_GENERATORS:
            raise ValueError(f"Unknown attack simulation: {name}")
        samples = ATTACK_GENERATORS[name](context, attempts, seed + index, realism=realism)
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
