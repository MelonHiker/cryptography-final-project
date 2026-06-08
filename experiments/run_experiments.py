"""End-to-end offline experiments for the report.

Produces, with zero hardware required:
  1. An ABLATION table: acoustic-only vs timing-only vs fusion (FAR/FRR/EER/AUC),
     proving the acoustic modality actually contributes.
  2. A ROC-curve figure overlaying the three feature views.
  3. A score-distribution histogram (owner vs imposter) for the fusion model.
  4. A "strictness" diagnostic comparing the single fused OC-SVM against the
     multi-gate model in modeling.py (the one that felt "too strict"): it reports
     how much owner data each one rejects (FRR), which explains the complaint.

Usage:
    # synthetic data (default, no hardware needed)
    python -m experiments.run_experiments

    # real collected data
    python -m experiments.run_experiments --owner-csv data/alice.csv \
        --imposter-csv data/classmates.csv

Outputs land in experiments/output/ (figures + results.md + results.csv).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs without a display
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from keystroke_auth.modeling import (
    ACOUSTIC_SLICE,
    TIMING_SLICE,
    evaluate_with_artifacts,
    load_feature_matrix,
    train_one_class_model,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# The three feature "views" we ablate over.
VIEWS: dict[str, slice] = {
    "timing_only": TIMING_SLICE,      # 8 timing features (cols 38:46)
    "acoustic_only": ACOUSTIC_SLICE,  # 38 acoustic features (cols 0:38)
    "fusion": slice(0, 46),           # all 46 features
}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _train_single_ocsvm(train: np.ndarray, nu: float, gamma) -> tuple[StandardScaler, OneClassSVM]:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(train)
    model = OneClassSVM(nu=nu, kernel="rbf", gamma=gamma)
    model.fit(scaled)
    return scaler, model


def _scores(scaler, model, matrix: np.ndarray) -> np.ndarray:
    return model.decision_function(scaler.transform(matrix)).reshape(-1)


def _equal_error_rate(owner: np.ndarray, imposter: np.ndarray) -> tuple[float, float]:
    """Return (EER, threshold_at_EER). Higher score == more 'owner-like'."""
    y_true = np.concatenate([np.ones_like(owner), np.zeros_like(imposter)])
    y_score = np.concatenate([owner, imposter])
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr  # FRR
    idx = int(np.nanargmin(np.abs(fpr - fnr)))  # FAR ~= FRR
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    return eer, float(thresholds[idx])


def evaluate_view(name: str, train: np.ndarray, owner_test: np.ndarray,
                  imposter_test: np.ndarray, nu: float, gamma) -> dict:
    sl = VIEWS[name]
    scaler, model = _train_single_ocsvm(train[:, sl], nu, gamma)
    owner_scores = _scores(scaler, model, owner_test[:, sl])
    imposter_scores = _scores(scaler, model, imposter_test[:, sl])

    auc = float(roc_auc_score(
        np.concatenate([np.ones_like(owner_scores), np.zeros_like(imposter_scores)]),
        np.concatenate([owner_scores, imposter_scores]),
    ))
    eer, thr = _equal_error_rate(owner_scores, imposter_scores)
    far = float(np.mean(imposter_scores > thr))   # imposters accepted
    frr = float(np.mean(owner_scores <= thr))      # owners rejected
    acc = float((np.sum(owner_scores > thr) + np.sum(imposter_scores <= thr))
                / (len(owner_scores) + len(imposter_scores)))
    return dict(view=name, auc=auc, eer=eer, far=far, frr=frr, accuracy=acc,
                threshold=thr, owner_scores=owner_scores, imposter_scores=imposter_scores)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_roc(results: list[dict], path: Path) -> None:
    plt.figure(figsize=(6, 6))
    for r in results:
        owner, imp = r["owner_scores"], r["imposter_scores"]
        y_true = np.concatenate([np.ones_like(owner), np.zeros_like(imp)])
        y_score = np.concatenate([owner, imp])
        fpr, tpr, _ = roc_curve(y_true, y_score)
        plt.plot(fpr, tpr, linewidth=2, label=f"{r['view']} (AUC={r['auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="random")
    plt.xlabel("False Acceptance Rate (FAR)")
    plt.ylabel("True Acceptance Rate (1 - FRR)")
    plt.title("ROC: ablation of feature modalities")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def plot_score_hist(fusion: dict, path: Path) -> None:
    plt.figure(figsize=(7, 4.5))
    plt.hist(fusion["owner_scores"], bins=20, alpha=0.6, label="owner", color="#2a9d8f")
    plt.hist(fusion["imposter_scores"], bins=20, alpha=0.6, label="imposter", color="#e76f51")
    plt.axvline(fusion["threshold"], color="black", linestyle="--",
                label=f"EER threshold={fusion['threshold']:.3f}")
    plt.xlabel("OC-SVM decision score (higher = more owner-like)")
    plt.ylabel("count")
    plt.title("Score distribution (fusion model)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


# --------------------------------------------------------------------------- #
# "Too strict" diagnostic
# --------------------------------------------------------------------------- #
def strictness_diagnostic(train: np.ndarray, owner_test: np.ndarray,
                          imposter_test: np.ndarray, nu: float, gamma) -> dict:
    """Compare the multi-gate fused model (modeling.train_one_class_model) against
    a plain single OC-SVM, reporting owner-rejection (FRR) for each."""
    artifacts = train_one_class_model(train, algorithm="oc_svm", nu=nu, gamma=gamma)

    gated_owner_rej = np.mean([
        not evaluate_with_artifacts(artifacts, r, threshold=-np.inf, strict=True)["accepted"]
        for r in owner_test
    ])
    gated_imp_acc = np.mean([
        evaluate_with_artifacts(artifacts, r, threshold=-np.inf, strict=True)["accepted"]
        for r in imposter_test
    ])
    loose_owner_rej = np.mean([
        not evaluate_with_artifacts(artifacts, r, threshold=-np.inf, strict=False)["accepted"]
        for r in owner_test
    ])

    # which gate fires most often on legitimate owners (strict mode)?
    gate_hits: dict[str, int] = {}
    for r in owner_test:
        for g in evaluate_with_artifacts(artifacts, r, threshold=-np.inf, strict=True)["failures"]:
            gate_hits[g] = gate_hits.get(g, 0) + 1

    return dict(gated_frr=float(gated_owner_rej), gated_far=float(gated_imp_acc),
                loose_frr=float(loose_owner_rej), gate_hits=gate_hits, n_owner=len(owner_test))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def load_or_synth(owner_csv: str | None, imposter_csv: str | None):
    if owner_csv and imposter_csv:
        owner = load_feature_matrix(owner_csv)
        imposter = load_feature_matrix(imposter_csv)
        if owner.shape[0] < 6 or imposter.shape[0] < 3:
            raise SystemExit("Need >=6 owner rows and >=3 imposter rows for a meaningful split.")
        n_train = max(int(owner.shape[0] * 0.6), 3)
        return owner[:n_train], owner[n_train:], imposter, "real CSV data"

    from experiments.synth import make_imposter, make_owner

    train = make_owner(40, seed=1)
    owner_test = make_owner(25, seed=2)
    imposter_test = make_imposter(25, seed=3)
    return train, owner_test, imposter_test, "synthetic data"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner-csv", default=None)
    parser.add_argument("--imposter-csv", default=None)
    parser.add_argument("--nu", type=float, default=0.05)
    parser.add_argument("--gamma", default="0.001")
    args = parser.parse_args()
    gamma = float(args.gamma) if _is_float(args.gamma) else args.gamma

    train, owner_test, imposter_test, source = load_or_synth(args.owner_csv, args.imposter_csv)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Data source: {source}")
    print(f"  train(owner)={train.shape[0]}  owner_test={owner_test.shape[0]}  "
          f"imposter_test={imposter_test.shape[0]}\n")

    results = [evaluate_view(v, train, owner_test, imposter_test, args.nu, gamma) for v in VIEWS]

    # --- ablation table ---
    header = f"{'view':>14} | {'AUC':>6} | {'EER':>7} | {'FAR':>7} | {'FRR':>7} | {'acc':>7}"
    print("=== Ablation (single OC-SVM per feature view) ===")
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['view']:>14} | {r['auc']:>6.3f} | {r['eer']*100:>6.1f}% | "
              f"{r['far']*100:>6.1f}% | {r['frr']*100:>6.1f}% | {r['accuracy']*100:>6.1f}%")

    # --- figures ---
    plot_roc(results, OUTPUT_DIR / "roc_ablation.png")
    fusion = next(r for r in results if r["view"] == "fusion")
    plot_score_hist(fusion, OUTPUT_DIR / "score_hist_fusion.png")

    # --- strictness diagnostic ---
    diag = strictness_diagnostic(train, owner_test, imposter_test, args.nu, gamma)
    print("\n=== Strict (multi-gate) vs Loose (single fusion) — owner rejection ===")
    print(f"  STRICT multi-gate FRR: {diag['gated_frr']*100:.1f}%   FAR: {diag['gated_far']*100:.1f}%")
    print(f"  LOOSE  single-gate FRR: {diag['loose_frr']*100:.1f}%")
    print(f"  Which gates reject legit owners (strict): {diag['gate_hits']}  (n={diag['n_owner']})")
    if diag["gated_frr"] > diag["loose_frr"] + 0.1:
        print("  -> Strict AND-logic compounds rejections into a high FRR. The product now"
              "\n     defaults to LOOSE mode (config.strict_mode=False); strict is a toggle.")

    # --- write results.md + csv ---
    _write_reports(results, fusion, diag, source)
    print(f"\nFigures + reports written to: {OUTPUT_DIR}")


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _write_reports(results, fusion, diag, source) -> None:
    with (OUTPUT_DIR / "results.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["view", "auc", "eer", "far", "frr", "accuracy", "threshold"])
        for r in results:
            w.writerow([r["view"], f"{r['auc']:.4f}", f"{r['eer']:.4f}", f"{r['far']:.4f}",
                        f"{r['frr']:.4f}", f"{r['accuracy']:.4f}", f"{r['threshold']:.4f}"])

    lines = [
        "# Experiment results", "",
        f"- Data source: **{source}**",
        f"- Owner test: {len(fusion['owner_scores'])}, "
        f"Imposter test: {len(fusion['imposter_scores'])}", "",
        "## Ablation (does the acoustic modality help?)", "",
        "| Feature view | AUC | EER | FAR | FRR | Accuracy |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(f"| {r['view']} | {r['auc']:.3f} | {r['eer']*100:.1f}% | "
                     f"{r['far']*100:.1f}% | {r['frr']*100:.1f}% | {r['accuracy']*100:.1f}% |")
    lines += [
        "", "Higher AUC and lower EER are better. If `fusion` beats both single",
        "modalities, the acoustic + timing combination is empirically justified.", "",
        "![ROC](roc_ablation.png)", "", "![Score histogram](score_hist_fusion.png)", "",
        "## Strict (multi-gate) vs Loose (single fusion)", "",
        f"- **Strict** multi-gate rejects **{diag['gated_frr']*100:.1f}%** of legitimate owners (FRR).",
        f"- **Loose** single-gate rejects only **{diag['loose_frr']*100:.1f}%** (this is the default).",
        f"- Gate rejections on owners (strict): `{diag['gate_hits']}`", "",
        "Strict mode requires the joint/acoustic/timing/balance gates to pass with",
        "AND-logic, so per-gate rejection probabilities compound into a high FRR.",
        "The product defaults to **loose** (`config.strict_mode = False`); strict is a",
        "toggle in the Authentication tab for high-security demos.",
    ]
    (OUTPUT_DIR / "results.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
