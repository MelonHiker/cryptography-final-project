"""Friendly one-window data collector for classmates.

Goal: let someone help you gather samples WITHOUT learning the full 6-stage
dashboard. One screen: type your name, press Start, then type the shown
passphrase a few times. Each sample's 46-D feature vector is appended to
``data/<name>.csv`` — the same format the training / experiment scripts read.

Run (inside the project venv, with a microphone connected):
    python collect.py

It reuses the real capture + feature pipeline, so the data is fully compatible
with dataset.csv and `python -m experiments.run_experiments --imposter-csv ...`.
"""

from __future__ import annotations

import re
from pathlib import Path
from threading import Thread

from keystroke_auth.capture import DependencyError, collect_enrollment_capture
from keystroke_auth.config import load_config
from keystroke_auth.features import extract_46_features
from keystroke_auth.modeling import append_feature_row

DATA_DIR = Path("data")


def _safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_\-]+", "_", name.strip()) or "anonymous"
    return cleaned.lower()


class CollectorApp:
    def __init__(self) -> None:
        import customtkinter as ctk  # imported here so a missing dep gives a clear message

        self.ctk = ctk
        self.config = load_config()
        self.passphrase = self.config.passphrase
        self.sample_rate = self.config.sample_rate
        self.key_length = self.config.key_length or 4500

        self.collected = 0
        self.target = 5
        self.busy = False

        ctk.set_appearance_mode("dark")
        self.root = ctk.CTk()
        self.root.title("Sample Collector")
        self.root.geometry("560x520")

        self._build_ui()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        ctk = self.ctk
        pad = dict(padx=24, pady=8)

        ctk.CTkLabel(
            self.root, text="Keystroke Sample Collector",
            font=ctk.CTkFont(size=22, weight="bold"),
        ).pack(padx=24, pady=(24, 4))
        ctk.CTkLabel(
            self.root,
            text="Help collect typing samples. Just type the phrase below\n"
                 "naturally each time it asks. Thank you!",
            justify="center",
        ).pack(padx=24, pady=(0, 12))

        ctk.CTkLabel(self.root, text="Your name / nickname:", anchor="w").pack(fill="x", **pad)
        self.name_var = ctk.StringVar(value="")
        ctk.CTkEntry(self.root, textvariable=self.name_var, width=480).pack(**pad)

        ctk.CTkLabel(self.root, text="How many samples:", anchor="w").pack(fill="x", **pad)
        self.count_var = ctk.StringVar(value="5")
        ctk.CTkEntry(self.root, textvariable=self.count_var, width=120).pack(anchor="w", padx=24)

        box = ctk.CTkFrame(self.root, corner_radius=14)
        box.pack(fill="x", padx=24, pady=16)
        ctk.CTkLabel(box, text="Type this phrase exactly:", anchor="w").pack(
            fill="x", padx=16, pady=(14, 2)
        )
        ctk.CTkLabel(
            box, text=self.passphrase,
            font=ctk.CTkFont(size=18, weight="bold"), text_color="#4cc2ff",
        ).pack(padx=16, pady=(0, 14))

        self.start_btn = ctk.CTkButton(
            self.root, text="Start", height=44, command=self.start
        )
        self.start_btn.pack(padx=24, pady=(4, 8))

        self.progress_var = ctk.StringVar(value="Ready.")
        ctk.CTkLabel(
            self.root, textvariable=self.progress_var,
            font=ctk.CTkFont(size=15), wraplength=500, justify="center",
        ).pack(padx=24, pady=(4, 20))

    # -------------------------------------------------------------- actions
    def start(self) -> None:
        if self.busy:
            return
        name = self.name_var.get().strip()
        if not name:
            self.progress_var.set("Please enter your name first.")
            return
        try:
            self.target = max(1, int(self.count_var.get().strip()))
        except ValueError:
            self.progress_var.set("Sample count must be a whole number.")
            return

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.output_path = DATA_DIR / f"{_safe_filename(name)}.csv"
        self.collected = 0
        self.busy = True
        self.start_btn.configure(state="disabled")
        self._next_sample()

    def _next_sample(self) -> None:
        if self.collected >= self.target:
            self._finish()
            return
        self.progress_var.set(
            f"Sample {self.collected + 1} / {self.target}\n"
            f"Start typing the phrase now (it records as you type)..."
        )
        Thread(target=self._capture_worker, daemon=True).start()

    def _capture_worker(self) -> None:
        try:
            audio, text_result = collect_enrollment_capture(self.sample_rate, self.passphrase)
            if text_result.invalidated:
                self.root.after(0, self._on_retry, "Phrase didn't match — let's retry that one.")
                return
            features = extract_46_features(
                audio, text_result.timestamps_sec, self.sample_rate, self.key_length
            )
            append_feature_row(self.output_path, features)
            self.root.after(0, self._on_sample_ok)
        except DependencyError as exc:
            self.root.after(0, self._on_fatal, f"Missing dependency: {exc}")
        except Exception as exc:  # noqa: BLE001 - surface any capture error to the user
            self.root.after(0, self._on_retry, f"Hiccup: {exc}. Retrying...")

    def _on_sample_ok(self) -> None:
        self.collected += 1
        self.root.after(400, self._next_sample)

    def _on_retry(self, message: str) -> None:
        self.progress_var.set(message)
        self.root.after(900, self._next_sample)

    def _on_fatal(self, message: str) -> None:
        self.busy = False
        self.start_btn.configure(state="normal")
        self.progress_var.set(message)

    def _finish(self) -> None:
        self.busy = False
        self.start_btn.configure(state="normal")
        self.progress_var.set(
            f"Done! Collected {self.collected} samples.\nSaved to: {self.output_path}"
        )

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    try:
        CollectorApp().run()
    except ImportError:
        raise SystemExit(
            "customtkinter is required. Install it with:  pip install customtkinter"
        )


if __name__ == "__main__":
    main()
