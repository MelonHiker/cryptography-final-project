from __future__ import annotations

import argparse

from .simulation import ATTACK_GENERATORS, format_result, run_simulation, save_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local attack simulations against a trained model.")
    parser.add_argument("--dataset", default="dataset.csv", help="Path to owner training dataset CSV.")
    parser.add_argument("--model", default="model.pkl", help="Path to model pickle.")
    parser.add_argument("--scaler", default="scaler.pkl", help="Path to scaler pickle.")
    parser.add_argument("--config", default="config.json", help="Path to config JSON.")
    parser.add_argument("--threshold", type=float, default=None, help="Override auth threshold.")
    parser.add_argument("--attempts", type=int, default=250, help="Attempts per attack.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--attack",
        action="append",
        choices=sorted(ATTACK_GENERATORS),
        help="Attack to run. Can be repeated. Defaults to all attacks.",
    )
    parser.add_argument("--output", help="Optional .json or .csv report output path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_simulation(
        dataset_path=args.dataset,
        model_path=args.model,
        scaler_path=args.scaler,
        config_path=args.config,
        attempts=args.attempts,
        threshold=args.threshold,
        seed=args.seed,
        attack_names=args.attack,
    )
    print(format_result(result))
    if args.output:
        save_result(result, args.output)


if __name__ == "__main__":
    main()

