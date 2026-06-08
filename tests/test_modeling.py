"""Unit tests for the modeling/scoring pipeline (modeling.py)."""

from __future__ import annotations

import numpy as np
import pytest

from keystroke_auth.modeling import (
    calibrate_auth_threshold,
    evaluate_with_artifacts,
    load_artifacts,
    load_feature_matrix,
    append_feature_row,
    save_artifacts,
    score_with_artifacts,
    train_one_class_model,
)


def _owner_matrix(n=40, seed=0):
    rng = np.random.default_rng(seed)
    # tight cluster = the legitimate owner
    return rng.normal(0.0, 1.0, size=(n, 46))


def test_train_rejects_wrong_shape():
    with pytest.raises(ValueError):
        train_one_class_model(np.zeros((10, 10)))


def test_train_rejects_empty():
    with pytest.raises(ValueError):
        train_one_class_model(np.zeros((0, 46)))


def test_owner_scores_higher_than_outlier():
    train = _owner_matrix()
    art = train_one_class_model(train, algorithm="oc_svm", nu=0.05, gamma=0.001)
    owner_score = score_with_artifacts(art, train[0])
    outlier = np.full(46, 50.0)  # far outside the training cluster
    outlier_score = score_with_artifacts(art, outlier)
    assert owner_score > outlier_score


def test_calibrate_threshold_separates_classes():
    owner = [1.0, 1.2, 0.9, 1.1, 1.05]
    imposter = [-1.0, -0.8, -1.2, -0.9, -1.1]
    thr, far, frr = calibrate_auth_threshold(owner, imposter)
    assert -1.0 < thr < 1.0
    assert far == 0.0 and frr == 0.0


def test_calibrate_handles_empty_input():
    assert calibrate_auth_threshold([], [1, 2, 3]) == (0.0, 0.0, 0.0)


def test_save_and_load_roundtrip(tmp_path):
    train = _owner_matrix()
    art = train_one_class_model(train, algorithm="oc_svm", nu=0.05, gamma=0.001)
    model_path = tmp_path / "model.pkl"
    scaler_path = tmp_path / "scaler.pkl"
    save_artifacts(art, model_path, scaler_path)

    reloaded = load_artifacts(model_path, scaler_path)
    before = score_with_artifacts(art, train[0])
    after = score_with_artifacts(reloaded, train[0])
    assert before == pytest.approx(after, abs=1e-9)


def test_evaluate_with_artifacts_returns_contract(tmp_path):
    train = _owner_matrix()
    art = train_one_class_model(train, algorithm="oc_svm", nu=0.05, gamma=0.001)
    result = evaluate_with_artifacts(art, train[0], threshold=-np.inf)
    assert set(result) >= {"accepted", "scores", "failures"}
    assert isinstance(result["accepted"], bool)
    assert "joint" in result["scores"]


def test_dataset_roundtrip_csv(tmp_path):
    path = tmp_path / "dataset.csv"
    row = list(range(46))
    append_feature_row(path, row)
    append_feature_row(path, row)
    matrix = load_feature_matrix(path)
    assert matrix.shape == (2, 46)
    np.testing.assert_allclose(matrix[0], row)
