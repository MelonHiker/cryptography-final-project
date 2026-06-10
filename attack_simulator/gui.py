from __future__ import annotations

from pathlib import Path
from threading import Thread

from .simulation import REALISM_PROFILES, format_result, run_simulation, save_result


class AttackSimulationApp:
    def __init__(self) -> None:
        import customtkinter as ctk

        self.ctk = ctk
        self.root = ctk.CTk()
        self.root.title("Attack Simulation & Validation")
        self.root.geometry("920x680")

        self.dataset_var = ctk.StringVar(value="dataset.csv")
        self.model_var = ctk.StringVar(value="model.pkl")
        self.scaler_var = ctk.StringVar(value="scaler.pkl")
        self.config_var = ctk.StringVar(value="config.json")
        self.threshold_var = ctk.StringVar(value="")
        self.attempts_var = ctk.StringVar(value="250")
        self.seed_var = ctk.StringVar(value="42")
        self.realism_var = ctk.StringVar(value="practical")
        self.output_var = ctk.StringVar(value="attack_simulation_report.json")
        self.status_var = ctk.StringVar(value="Ready.")
        self.result_text = None

    def launch(self) -> None:
        ctk = self.ctk
        ctk.set_appearance_mode("system")
        ctk.set_default_color_theme("dark-blue")

        header = ctk.CTkFrame(self.root, corner_radius=8)
        header.pack(fill="x", padx=18, pady=(18, 10))
        ctk.CTkLabel(
            header,
            text="Attack Simulation & Validation",
            font=ctk.CTkFont(size=24, weight="bold"),
        ).pack(anchor="w", padx=16, pady=(14, 4))
        ctk.CTkLabel(header, textvariable=self.status_var).pack(
            anchor="w", padx=16, pady=(0, 14)
        )

        body = ctk.CTkFrame(self.root, corner_radius=8)
        body.pack(fill="both", expand=True, padx=18, pady=(0, 18))

        form = ctk.CTkFrame(body, fg_color="transparent")
        form.pack(fill="x", padx=16, pady=16)

        fields = [
            ("Dataset CSV", self.dataset_var),
            ("Model PKL", self.model_var),
            ("Scaler PKL", self.scaler_var),
            ("Config JSON", self.config_var),
            ("Threshold Override", self.threshold_var),
            ("Attempts / Attack", self.attempts_var),
            ("Seed", self.seed_var),
            ("Report Output", self.output_var),
        ]
        for index, (label, variable) in enumerate(fields):
            row = index // 2
            column = index % 2
            frame = ctk.CTkFrame(form, fg_color="transparent")
            frame.grid(row=row, column=column, sticky="ew", padx=(0, 16), pady=(0, 10))
            ctk.CTkLabel(frame, text=label, anchor="w").pack(anchor="w")
            ctk.CTkEntry(frame, textvariable=variable, width=360).pack(fill="x")
        form.grid_columnconfigure(0, weight=1)
        form.grid_columnconfigure(1, weight=1)

        realism_frame = ctk.CTkFrame(body, fg_color="transparent")
        realism_frame.pack(fill="x", padx=16, pady=(0, 12))
        ctk.CTkLabel(realism_frame, text="Attack Realism", anchor="w").pack(side="left")
        ctk.CTkOptionMenu(
            realism_frame,
            variable=self.realism_var,
            values=list(sorted(REALISM_PROFILES)),
            width=180,
        ).pack(side="left", padx=(10, 0))

        actions = ctk.CTkFrame(body, fg_color="transparent")
        actions.pack(fill="x", padx=16, pady=(0, 12))
        ctk.CTkButton(actions, text="Run Simulations", command=self._run_clicked).pack(side="left")
        ctk.CTkButton(actions, text="Save Report", command=self._save_clicked).pack(
            side="left", padx=(10, 0)
        )

        self.result_text = ctk.CTkTextbox(body, font=ctk.CTkFont(family="Courier", size=13))
        self.result_text.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        self.result_text.insert(
            "end",
            "Run local simulations after dataset.csv, model.pkl, and scaler.pkl are available.",
        )

        self.root.mainloop()

    def _set_result(self, text: str) -> None:
        if self.result_text is None:
            return
        self.result_text.configure(state="normal")
        self.result_text.delete("1.0", "end")
        self.result_text.insert("end", text)
        self.result_text.configure(state="disabled")

    def _build_result(self):
        threshold_text = self.threshold_var.get().strip()
        threshold = float(threshold_text) if threshold_text else None
        return run_simulation(
            dataset_path=self.dataset_var.get().strip(),
            model_path=self.model_var.get().strip(),
            scaler_path=self.scaler_var.get().strip(),
            config_path=self.config_var.get().strip(),
            threshold=threshold,
            attempts=int(self.attempts_var.get().strip()),
            seed=int(self.seed_var.get().strip()),
            realism=self.realism_var.get().strip(),
        )

    def _run_clicked(self) -> None:
        self.status_var.set("Running simulations...")

        def worker() -> None:
            try:
                result = self._build_result()
                text = format_result(result)
                self.root.after(0, lambda: self._set_result(text))
                self.root.after(0, lambda: self.status_var.set("Simulation complete."))
                self.last_result = result
            except Exception as exc:
                message = f"Simulation failed:\n{exc}"
                self.root.after(0, lambda: self._set_result(message))
                self.root.after(0, lambda: self.status_var.set("Simulation failed."))

        Thread(target=worker, daemon=True).start()

    def _save_clicked(self) -> None:
        try:
            result = getattr(self, "last_result", None)
            if result is None:
                result = self._build_result()
                self.last_result = result
            output_path = Path(self.output_var.get().strip() or "attack_simulation_report.json")
            save_result(result, output_path)
            self.status_var.set(f"Saved report to {output_path}")
        except Exception as exc:
            self.status_var.set(f"Save failed: {exc}")


def launch_app() -> None:
    AttackSimulationApp().launch()
