from __future__ import annotations

import csv
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [f"f{i}" for i in range(1, 47)]


@dataclass(slots=True)
class OneClassArtifacts:
    scaler: StandardScaler
    model: Any
    feature_columns: list[str]


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

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    algo_lower = algorithm.lower()

    if algo_lower == "lof":
        from sklearn.neighbors import LocalOutlierFactor

        n_neighbors = min(5, features.shape[0] - 1)
        if n_neighbors < 1:
            n_neighbors = 1
        contamination = max(min(nu, 0.5), 0.001)
        model = LocalOutlierFactor(
            n_neighbors=n_neighbors, novelty=True, contamination=contamination
        )
    elif algo_lower == "iforest":
        from sklearn.ensemble import IsolationForest

        contamination = max(min(nu, 0.5), 0.001)
        model = IsolationForest(
            n_estimators=n_estimators, contamination=contamination, random_state=42
        )
    elif algo_lower == "pca_svm":
        from sklearn.decomposition import PCA
        from sklearn.pipeline import make_pipeline
        from sklearn.svm import OneClassSVM

        actual_components = min(n_components, features.shape[0], features.shape[1])
        if actual_components < 1:
            actual_components = 1
        model = make_pipeline(
            PCA(n_components=actual_components),
            OneClassSVM(nu=nu, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0),
        )
    else:
        from sklearn.svm import OneClassSVM

        model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0)

    model.fit(scaled)
    return OneClassArtifacts(scaler=scaler, model=model, feature_columns=FEATURE_COLUMNS.copy())


def score_with_artifacts(artifacts: OneClassArtifacts, feature_vector: Sequence[float]) -> float:
    features = np.asarray(feature_vector, dtype=np.float64).reshape(1, -1)
    scaled = artifacts.scaler.transform(features)
    if not hasattr(artifacts.model, "decision_function"):
        raise AttributeError("Loaded model does not support decision_function")
    return float(np.asarray(artifacts.model.decision_function(scaled)).reshape(-1)[0])


def predict_with_artifacts(
    artifacts: OneClassArtifacts,
    feature_vector: Sequence[float],
    threshold: float = 0.0,
) -> int:
    return 1 if score_with_artifacts(artifacts, feature_vector) > threshold else -1


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
            {"model": artifacts.model, "feature_columns": artifacts.feature_columns}, handle
        )
    with Path(scaler_path).open("wb") as handle:
        pickle.dump(artifacts.scaler, handle)


def load_artifacts(model_path: str | Path, scaler_path: str | Path) -> OneClassArtifacts:
    with Path(model_path).open("rb") as handle:
        model_payload = pickle.load(handle)
    with Path(scaler_path).open("rb") as handle:
        scaler = pickle.load(handle)

    if isinstance(model_payload, dict):
        model = model_payload["model"]
        feature_columns = model_payload.get("feature_columns", FEATURE_COLUMNS.copy())
    else:
        model = model_payload
        feature_columns = FEATURE_COLUMNS.copy()

    return OneClassArtifacts(scaler=scaler, model=model, feature_columns=feature_columns)
