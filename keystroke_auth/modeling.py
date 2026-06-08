from __future__ import annotations

import csv
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [f"f{i}" for i in range(1, 47)]
ACOUSTIC_SLICE = slice(0, 38)
TIMING_SLICE = slice(38, 46)


@dataclass(slots=True)
class OneClassArtifacts:
    scaler: StandardScaler
    model: Any
    feature_columns: list[str]
    acoustic_scaler: StandardScaler | None = None
    acoustic_model: Any | None = None
    timing_scaler: StandardScaler | None = None
    timing_model: Any | None = None
    score_thresholds: dict[str, float] = field(default_factory=dict)
    score_stats: dict[str, tuple[float, float]] = field(default_factory=dict)


def append_feature_row(dataset_path: str | Path, features: Sequence[float]) -> None:
    path = Path(dataset_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if is_new:
            writer.writerow(FEATURE_COLUMNS)
        writer.writerow([float(value) for value in features])


def load_feature_matrix(dataset_path: str | Path) -> np.ndarray:
    path = Path(dataset_path)
    if not path.exists():
        return np.zeros((0, 46), dtype=np.float64)

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        rows = list(reader)

    if not rows:
        return np.zeros((0, 46), dtype=np.float64)

    if rows[0] == FEATURE_COLUMNS:
        rows = rows[1:]

    matrix = np.asarray(
        [[float(value) for value in row[:46]] for row in rows if row], dtype=np.float64
    )
    if matrix.size == 0:
        return np.zeros((0, 46), dtype=np.float64)
    return matrix.reshape(-1, 46)


def _build_estimator(
    features: np.ndarray,
    algorithm: str,
    nu: float,
    kernel: str,
    gamma: str | float,
    degree: int,
    coef0: float,
    n_estimators: int,
    n_components: int,
):
    algo_lower = algorithm.lower()

    if algo_lower == "lof":
        from sklearn.neighbors import LocalOutlierFactor

        n_neighbors = min(5, features.shape[0] - 1)
        if n_neighbors < 1:
            n_neighbors = 1
        contamination = max(min(nu, 0.5), 0.001)
        return LocalOutlierFactor(
            n_neighbors=n_neighbors, novelty=True, contamination=contamination
        )

    if algo_lower == "iforest":
        from sklearn.ensemble import IsolationForest

        contamination = max(min(nu, 0.5), 0.001)
        return IsolationForest(
            n_estimators=n_estimators, contamination=contamination, random_state=42
        )

    if algo_lower == "pca_svm":
        from sklearn.decomposition import PCA
        from sklearn.pipeline import make_pipeline
        from sklearn.svm import OneClassSVM

        actual_components = min(n_components, features.shape[0], features.shape[1])
        if actual_components < 1:
            actual_components = 1
        return make_pipeline(
            PCA(n_components=actual_components),
            OneClassSVM(nu=nu, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0),
        )

    from sklearn.svm import OneClassSVM

    return OneClassSVM(nu=nu, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0)


def _fit_view_model(
    features: np.ndarray,
    algorithm: str,
    nu: float,
    kernel: str,
    gamma: str | float,
    degree: int,
    coef0: float,
    n_estimators: int,
    n_components: int,
) -> tuple[StandardScaler, Any]:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    model = _build_estimator(
        features,
        algorithm,
        nu,
        kernel,
        gamma,
        degree,
        coef0,
        n_estimators,
        n_components,
    )
    model.fit(scaled)
    return scaler, model


def _score_with_model(
    scaler: StandardScaler,
    model: Any,
    feature_vector: Sequence[float] | np.ndarray,
) -> float:
    features = np.asarray(feature_vector, dtype=np.float64).reshape(1, -1)
    scaled = scaler.transform(features)
    if not hasattr(model, "decision_function"):
        raise AttributeError("Loaded model does not support decision_function")
    return float(np.asarray(model.decision_function(scaled)).reshape(-1)[0])


def _score_matrix_with_model(
    scaler: StandardScaler,
    model: Any,
    matrix: np.ndarray,
) -> np.ndarray:
    if not hasattr(model, "decision_function"):
        raise AttributeError("Loaded model does not support decision_function")
    scaled = scaler.transform(matrix)
    return np.asarray(model.decision_function(scaled), dtype=np.float64).reshape(-1)


def _score_floor(scores: np.ndarray, quantile: float) -> float:
    if scores.size == 0:
        return 0.0
    return float(np.quantile(scores, quantile))


def _score_stat_pair(scores: np.ndarray) -> tuple[float, float]:
    if scores.size == 0:
        return 0.0, 1.0
    mean = float(np.mean(scores))
    std = float(np.std(scores, ddof=0))
    return mean, max(std, 1e-6)


def _zscore(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (values - mean) / max(std, 1e-6)


def train_one_class_model(
    features: np.ndarray,
    algorithm: str = "lof",
    nu: float = 0.1,
    kernel: str = "rbf",
    gamma: str | float = "scale",
    degree: int = 3,
    coef0: float = 0.0,
    n_estimators: int = 100,
    n_components: int = 5,
) -> OneClassArtifacts:
    if features.ndim != 2 or features.shape[1] != 46:
        raise ValueError("Expected a 2D matrix with 46 feature columns")
    if features.shape[0] == 0:
        raise ValueError("Training data is empty")

    scaler, model = _fit_view_model(
        features, algorithm, nu, kernel, gamma, degree, coef0, n_estimators, n_components
    )
    acoustic_scaler, acoustic_model = _fit_view_model(
        features[:, ACOUSTIC_SLICE],
        algorithm,
        nu,
        kernel,
        gamma,
        degree,
        coef0,
        n_estimators,
        n_components,
    )
    timing_scaler, timing_model = _fit_view_model(
        features[:, TIMING_SLICE],
        algorithm,
        nu,
        kernel,
        gamma,
        degree,
        coef0,
        n_estimators,
        n_components,
    )

    joint_scores = _score_matrix_with_model(scaler, model, features)
    acoustic_scores = _score_matrix_with_model(
        acoustic_scaler, acoustic_model, features[:, ACOUSTIC_SLICE]
    )
    timing_scores = _score_matrix_with_model(timing_scaler, timing_model, features[:, TIMING_SLICE])

    acoustic_stats = _score_stat_pair(acoustic_scores)
    timing_stats = _score_stat_pair(timing_scores)
    balance_scores = np.abs(
        _zscore(acoustic_scores, acoustic_stats[0], acoustic_stats[1])
        - _zscore(timing_scores, timing_stats[0], timing_stats[1])
    )

    score_thresholds = {
        "joint": _score_floor(joint_scores, 0.05),
        "acoustic": _score_floor(acoustic_scores, 0.12),
        "timing": _score_floor(timing_scores, 0.12),
        "balance_max": float(np.quantile(balance_scores, 0.9)),
    }
    score_stats = {
        "joint": _score_stat_pair(joint_scores),
        "acoustic": acoustic_stats,
        "timing": timing_stats,
    }

    return OneClassArtifacts(
        scaler=scaler,
        model=model,
        feature_columns=FEATURE_COLUMNS.copy(),
        acoustic_scaler=acoustic_scaler,
        acoustic_model=acoustic_model,
        timing_scaler=timing_scaler,
        timing_model=timing_model,
        score_thresholds=score_thresholds,
        score_stats=score_stats,
    )


def score_with_artifacts(artifacts: OneClassArtifacts, feature_vector: Sequence[float]) -> float:
    return _score_with_model(artifacts.scaler, artifacts.model, feature_vector)


def score_breakdown_with_artifacts(
    artifacts: OneClassArtifacts, feature_vector: Sequence[float]
) -> dict[str, float]:
    features = np.asarray(feature_vector, dtype=np.float64).reshape(-1)
    scores = {"joint": score_with_artifacts(artifacts, features)}

    if artifacts.acoustic_scaler is not None and artifacts.acoustic_model is not None:
        scores["acoustic"] = _score_with_model(
            artifacts.acoustic_scaler, artifacts.acoustic_model, features[ACOUSTIC_SLICE]
        )

    if artifacts.timing_scaler is not None and artifacts.timing_model is not None:
        scores["timing"] = _score_with_model(
            artifacts.timing_scaler, artifacts.timing_model, features[TIMING_SLICE]
        )

    if "acoustic" in scores and "timing" in scores:
        acoustic_mean, acoustic_std = artifacts.score_stats.get("acoustic", (0.0, 1.0))
        timing_mean, timing_std = artifacts.score_stats.get("timing", (0.0, 1.0))
        acoustic_z = (scores["acoustic"] - acoustic_mean) / max(acoustic_std, 1e-6)
        timing_z = (scores["timing"] - timing_mean) / max(timing_std, 1e-6)
        scores["balance"] = abs(acoustic_z - timing_z)

    return scores


def evaluate_with_artifacts(
    artifacts: OneClassArtifacts,
    feature_vector: Sequence[float],
    threshold: float = 0.0,
) -> dict[str, Any]:
    scores = score_breakdown_with_artifacts(artifacts, feature_vector)
    active_thresholds = {
        "joint": max(float(threshold), float(artifacts.score_thresholds.get("joint", threshold)))
    }
    failures: list[str] = []

    if scores["joint"] <= active_thresholds["joint"]:
        failures.append("joint")

    if "acoustic" in scores:
        active_thresholds["acoustic"] = float(artifacts.score_thresholds.get("acoustic", -np.inf))
        if scores["acoustic"] <= active_thresholds["acoustic"]:
            failures.append("acoustic")

    if "timing" in scores:
        active_thresholds["timing"] = float(artifacts.score_thresholds.get("timing", -np.inf))
        if scores["timing"] <= active_thresholds["timing"]:
            failures.append("timing")

    if "balance" in scores and "balance_max" in artifacts.score_thresholds:
        active_thresholds["balance_max"] = float(artifacts.score_thresholds["balance_max"])
        if scores["balance"] > active_thresholds["balance_max"]:
            failures.append("balance")

    return {
        "accepted": not failures,
        "scores": scores,
        "thresholds": active_thresholds,
        "failures": failures,
    }


def predict_with_artifacts(
    artifacts: OneClassArtifacts,
    feature_vector: Sequence[float],
    threshold: float = 0.0,
) -> int:
    evaluation = evaluate_with_artifacts(artifacts, feature_vector, threshold=threshold)
    return 1 if evaluation["accepted"] else -1


def calibrate_auth_threshold(
    owner_scores: Sequence[float] | np.ndarray,
    imposter_scores: Sequence[float] | np.ndarray,
) -> tuple[float, float, float]:
    owner = np.asarray(owner_scores, dtype=np.float64).reshape(-1)
    imposter = np.asarray(imposter_scores, dtype=np.float64).reshape(-1)

    if owner.size == 0 or imposter.size == 0:
        return 0.0, 0.0, 0.0

    score_pool = np.unique(np.concatenate([owner, imposter]))
    if score_pool.size == 1:
        threshold = float(score_pool[0])
        far = float(np.mean(imposter > threshold))
        frr = float(np.mean(owner <= threshold))
        return threshold, far, frr

    candidates = np.concatenate(
        [
            np.array([score_pool[0] - 1e-9], dtype=np.float64),
            (score_pool[:-1] + score_pool[1:]) / 2.0,
            np.array([score_pool[-1] + 1e-9], dtype=np.float64),
        ]
    )

    best_threshold = float(candidates[0])
    best_far = float(np.mean(imposter > best_threshold))
    best_frr = float(np.mean(owner <= best_threshold))
    best_gap = abs(best_far - best_frr)
    best_error = best_far + best_frr

    for threshold in candidates[1:]:
        far = float(np.mean(imposter > threshold))
        frr = float(np.mean(owner <= threshold))
        gap = abs(far - frr)
        error = far + frr
        if (gap < best_gap) or (np.isclose(gap, best_gap) and error < best_error):
            best_threshold = float(threshold)
            best_far = far
            best_frr = frr
            best_gap = gap
            best_error = error

    return best_threshold, best_far, best_frr


def save_artifacts(
    artifacts: OneClassArtifacts, model_path: str | Path, scaler_path: str | Path
) -> None:
    with Path(model_path).open("wb") as handle:
        pickle.dump(
            {
                "model": artifacts.model,
                "acoustic_model": artifacts.acoustic_model,
                "timing_model": artifacts.timing_model,
                "feature_columns": artifacts.feature_columns,
                "score_thresholds": artifacts.score_thresholds,
                "score_stats": artifacts.score_stats,
            },
            handle,
        )
    with Path(scaler_path).open("wb") as handle:
        pickle.dump(
            {
                "joint": artifacts.scaler,
                "acoustic": artifacts.acoustic_scaler,
                "timing": artifacts.timing_scaler,
            },
            handle,
        )


def load_artifacts(model_path: str | Path, scaler_path: str | Path) -> OneClassArtifacts:
    with Path(model_path).open("rb") as handle:
        model_payload = pickle.load(handle)
    with Path(scaler_path).open("rb") as handle:
        scaler = pickle.load(handle)

    if isinstance(model_payload, dict):
        model = model_payload["model"]
        feature_columns = model_payload.get("feature_columns", FEATURE_COLUMNS.copy())
        acoustic_model = model_payload.get("acoustic_model")
        timing_model = model_payload.get("timing_model")
        score_thresholds = {
            str(key): float(value)
            for key, value in model_payload.get("score_thresholds", {}).items()
        }
        score_stats = {
            str(key): (float(value[0]), float(value[1]))
            for key, value in model_payload.get("score_stats", {}).items()
        }
    else:
        model = model_payload
        feature_columns = FEATURE_COLUMNS.copy()
        acoustic_model = None
        timing_model = None
        score_thresholds = {}
        score_stats = {}

    if isinstance(scaler, dict):
        joint_scaler = scaler.get("joint")
        acoustic_scaler = scaler.get("acoustic")
        timing_scaler = scaler.get("timing")
    else:
        joint_scaler = scaler
        acoustic_scaler = None
        timing_scaler = None

    return OneClassArtifacts(
        scaler=joint_scaler,
        model=model,
        feature_columns=feature_columns,
        acoustic_scaler=acoustic_scaler,
        acoustic_model=acoustic_model,
        timing_scaler=timing_scaler,
        timing_model=timing_model,
        score_thresholds=score_thresholds,
        score_stats=score_stats,
    )
