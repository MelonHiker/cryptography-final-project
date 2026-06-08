"""Recordable external-attack demo (the 'Hydra' video).

Unlike simulation.py (which fabricates feature vectors), this driver runs the
REAL feature-extraction + scoring pipeline against automated login attempts,
printing a live, screen-recordable log. Use it for the 'external attack' demo
video: an attacker who KNOWS the password but is a bot still gets blocked,
triggering the OTP fallback.

Two modes:

  synthetic (default, no hardware -- always works for recording):
      Builds mechanical-timing keystrokes and a silent microphone track (a bot
      makes no real key-click sound), then runs the actual extract_46_features +
      model decision. Demonstrates why a password-correct bot is rejected.

  live (truly end-to-end, needs a mic + the attacked window focused):
      Uses pynput to INJECT keystrokes at machine speed while the real audio +
      keystroke capture runs, then scores the captured sample. Click an empty
      text field within the 3s countdown so the injected text lands there.

Examples:
    python -m attack_simulator.attack_demo                 # synthetic, 15 tries
    python -m attack_simulator.attack_demo --attempts 30
    python -m attack_simulator.attack_demo --mode live --attempts 5
    python -m attack_simulator.attack_demo --wordlist rockyou_top.txt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from keystroke_auth.config import load_config
from keystroke_auth.features import extract_46_features
from keystroke_auth.modeling import (
    evaluate_with_artifacts,
    load_artifacts,
    train_one_class_model,
)

# A few common passwords; the attack also always tries the REAL passphrase to
# prove that even a correct guess is blocked by the behavioural layer.
DEFAULT_WORDLIST = [
    "123456", "password", "qwerty", "letmein", "admin123",
    "iloveyou", "000000", "dragon", "monkey", "football",
]

GREEN, RED, CYAN, DIM, BOLD, RESET = (
    "\033[92m", "\033[91m", "\033[96m", "\033[2m", "\033[1m", "\033[0m",
)


def _load_or_make_model(config):
    """Load the enrolled model, or train a throwaway demo model so the video
    runs even before real enrollment."""
    model_path, scaler_path = config.model_path, config.scaler_path
    if Path(model_path).exists() and Path(scaler_path).exists():
        print(f"{DIM}Loaded enrolled model: {model_path}{RESET}")
        return load_artifacts(model_path, scaler_path), False
    print(f"{DIM}No enrolled model found - training a DEMO owner model "
          f"(synthetic).{RESET}")
    from experiments.synth import make_owner

    artifacts = train_one_class_model(make_owner(40, seed=1), algorithm="oc_svm",
                                      nu=0.05, gamma=0.001)
    return artifacts, True


def _mechanical_sample(passphrase: str, sample_rate: int, key_length: int):
    """A bot: near-constant tiny inter-key gaps and a SILENT mic track."""
    n_keys = max(len(passphrase), 2)
    t = 0.5
    timestamps = []
    for _ in range(n_keys):
        timestamps.append(t)
        t += 0.008 + np.random.uniform(0, 0.002)  # ~8 ms, machine-like
    duration = timestamps[-1] + 0.5
    audio = np.random.normal(0, 0.0002, size=int(duration * sample_rate))  # silence
    return audio, np.array(timestamps)


def _live_sample(passphrase: str, sample_rate: int, key_length: int):
    from pynput.keyboard import Controller

    from keystroke_auth.capture import collect_enrollment_capture

    print(f"{CYAN}  Injecting keystrokes in 3s - click an empty text field NOW...{RESET}")
    time.sleep(3)
    result = {}

    import threading

    def _attacker():
        time.sleep(0.6)
        Controller().type(passphrase)  # machine-speed injection

    threading.Thread(target=_attacker, daemon=True).start()
    audio, text_result = collect_enrollment_capture(sample_rate, passphrase)
    result["audio"] = audio
    result["timestamps"] = np.array(text_result.timestamps_sec)
    return result["audio"], result["timestamps"]


def run(mode: str, attempts: int, wordlist: list[str], strict: bool, delay: float) -> None:
    config = load_config()
    artifacts, demo_model = _load_or_make_model(config)
    threshold = float(config.auth_threshold)
    passphrase = config.passphrase
    sr, kl = config.sample_rate, config.key_length or 4500

    # always include the correct passphrase among the guesses
    guesses = (wordlist * ((attempts // len(wordlist)) + 1))[:attempts - 1] + [passphrase]

    print(f"\n{BOLD}=== EXTERNAL ATTACK DEMO ==={RESET}")
    print(f"Target passphrase: {CYAN}{passphrase!r}{RESET}")
    print(f"Mode: {mode}   Decision: {'STRICT multi-gate' if strict else 'LOOSE single-fusion'}"
          f"   Threshold: {threshold:.4f}\n")

    accepted = 0
    for i, guess in enumerate(guesses, 1):
        if mode == "live":
            audio, ts = _live_sample(passphrase, sr, kl)
        else:
            audio, ts = _mechanical_sample(passphrase, sr, kl)

        features = extract_46_features(audio, ts, sr, kl)
        result = evaluate_with_artifacts(artifacts, features, threshold=threshold, strict=strict)
        ok = result["accepted"]
        score = result["scores"]["joint"]
        is_real = guess == passphrase

        if ok:
            accepted += 1
            verdict = f"{GREEN}ACCEPTED{RESET}"
            note = f"{RED}<-- BREACH{RESET}"
        else:
            verdict = f"{RED}REJECTED{RESET} -> OTP sent to owner (login blocked)"
            note = ""
        tag = f"{CYAN}[correct password]{RESET}" if is_real else ""
        print(f"  #{i:02d} try={guess:<12.12} score={score:+.3f}  {verdict} {tag}{note}")
        if delay:
            time.sleep(delay)

    blocked = attempts - accepted
    print(f"\n{BOLD}=== SUMMARY ==={RESET}")
    print(f"  Attempts:      {attempts}")
    print(f"  Blocked:       {GREEN}{blocked}/{attempts} ({blocked/attempts*100:.1f}%){RESET}")
    print(f"  Got through:   {RED}{accepted}{RESET}")
    print(f"\n  The behavioural-biometric layer rejects automated attempts even when the\n"
          f"  password is correct, because a bot cannot reproduce the owner's typing\n"
          f"  rhythm and key-press sound. Each rejection triggers the email OTP fallback.")
    if demo_model:
        print(f"\n{DIM}  (Demo model used. Enroll + train for results on your real profile.){RESET}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["synthetic", "live"], default="synthetic")
    parser.add_argument("--attempts", type=int, default=15)
    parser.add_argument("--wordlist", default=None, help="file with one password per line")
    parser.add_argument("--strict", action="store_true", help="use strict multi-gate decision")
    parser.add_argument("--delay", type=float, default=0.25, help="seconds between tries (visual)")
    args = parser.parse_args()

    wordlist = DEFAULT_WORDLIST
    if args.wordlist and Path(args.wordlist).exists():
        wordlist = [w.strip() for w in Path(args.wordlist).read_text(encoding="utf-8").splitlines() if w.strip()]

    run(args.mode, max(args.attempts, 1), wordlist, args.strict, args.delay)


if __name__ == "__main__":
    main()
