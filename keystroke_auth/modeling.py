from __future__ import annotations

import csv
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

FEATURE_COLUMNS = [f"f{i}" for i in range(1, 47)]


@dataclass(slots=True)
class OneClassArtifacts:
    scaler: StandardScaler
    model: OneClassSVM
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
    nu: float = 0.1,
    kernel: str = "rbf",
    gamma: str | float = "scale",
    degree: int = 3,
    coef0: float = 0.0,
) -> OneClassArtifacts:
    if features.ndim != 2 or features.shape[1] != 46:
        raise ValueError("Expected a 2D matrix with 46 feature columns")
    if features.shape[0] == 0:
        raise ValueError("Training data is empty")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0)
    model.fit(scaled)
    return OneClassArtifacts(scaler=scaler, model=model, feature_columns=FEATURE_COLUMNS.copy())


def predict_with_artifacts(artifacts: OneClassArtifacts, feature_vector: Sequence[float]) -> int:
    features = np.asarray(feature_vector, dtype=np.float64).reshape(1, -1)
    scaled = artifacts.scaler.transform(features)
    return int(artifacts.model.predict(scaled)[0])


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
