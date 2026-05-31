from __future__ import annotations

from pathlib import Path
from threading import Thread
from time import perf_counter

import numpy as np

from .calibration import estimate_key_length, persist_calibration
from .capture import DependencyError
from .config import load_config, save_config
from .features import extract_46_features
from .modeling import (
    append_feature_row,
    calibrate_auth_threshold,
    load_feature_matrix,
    save_artifacts,
    train_one_class_model,
)
from .otp import generate_otp, send_otp_email


class KeystrokeAuthApp:
    def __init__(self, config_path: str | Path = "config.json") -> None:
        self.config_path = Path(config_path)
        self.config = load_config(self.config_path)
        self.root = None
        self.status_var = None
        self.passphrase_var = None
        self.otp_var = None
        self.email_var = None
        self.model_loaded = None
        self.pending_otp = None
        self.feature_preview_var = None
        self.feature_output = None
        self.calibration_active = False
        self.calibration_press_target = 3
        self.calibration_press_times: list[float] = []
        self.calibration_stop_event = None
        self.calibration_thread = None
        self.calibration_notice_var = None
        self.tab_view = None
        self.collection_target_var = None
        self.collection_dialog = None
        self.collection_progress_var = None
        self.collection_prompt_var = None
        self.collection_input_var = None
        self.collection_input_entry = None
        self.collection_expected_count = 100
        self.collection_current_index = 0
        self.collection_active = False
        self.collection_capture_audio = None
        self.collection_capture_stop_event = None
        self.collection_capture_thread = None
        self.collection_capture_started_at = 0.0
        self.collection_capture_timestamps: list[float] = []
        self.collection_capture_previous_length = 0
        self.collection_expected_text = ""
        self.collection_key_bind_id = None
        self.collection_session_plan: list[int] = []
        self.collection_session_index = 0
        self.collection_session_sample_count = 0
        self.collection_session_count = 4
        self.collection_session_paused = False
        self.collection_continue_button = None
        self.collection_noise_std = 0.0015
        self.model_dataset_path_var = None
        self.model_nu_var = None
        self.model_gamma_var = None
        self.auth_gmail_var = None
        self.auth_passphrase_var = None
        self.auth_otp_var = None
        self.auth_model_path_var = None
        self.auth_gmail_dialog = None
        self.auth_passphrase_dialog = None
        self.otp_frame = None
        self.success_frame = None
        self.login_success_var = None
        self.auth_otp_entry = None
        self.capture_active = False
        self.capture_expected_text = ""
        self.capture_mode = ""
        self.capture_timestamps: list[float] = []
        self.capture_previous_length = 0
        self.capture_started_at = 0.0
        self.capture_audio = None
        self.capture_audio_stop_event = None
        self.capture_audio_thread = None
        self.capture_success_callback = None
        self.capture_error_callback = None

    def _require_ctk(self):
        try:
            import customtkinter as ctk  # type: ignore
        except ImportError as exc:
            raise DependencyError("customtkinter is required to launch the GUI") from exc
        return ctk

    def _set_status(self, message: str) -> None:
        if self.status_var is not None:
            self.status_var.set(message)

    def _run_background(self, callback, on_success=None, on_error=None) -> None:
        import queue

        q = queue.Queue()

        def worker() -> None:
            try:
                result = callback()
                q.put(("success", result))
            except BaseException as exc:
                q.put(("error", exc))

        def check_queue():
            try:
                status, payload = q.get_nowait()
                if status == "success" and on_success:
                    on_success(payload)
                elif status == "error":
                    if on_error:
                        on_error(payload)
                    else:
                        self._set_status(str(payload))
            except queue.Empty:
                self.root.after(50, check_queue)

        self.root.after(50, check_queue)
        Thread(target=worker, daemon=True).start()

    def launch(self) -> None:
        ctk = self._require_ctk()
        ctk.set_appearance_mode("system")
        ctk.set_default_color_theme("dark-blue")

        self.root = ctk.CTk()
        self.root.title("Keystroke Authentication Lab")
        self.root.geometry("1080x720")

        self.status_var = ctk.StringVar(value="Ready.")
        self.passphrase_var = ctk.StringVar(value=self.config.passphrase)
        self.otp_var = ctk.StringVar(value="")
        self.email_var = ctk.StringVar(value=self.config.smtp_recipient)
        self.feature_preview_var = ctk.StringVar(value="No features extracted yet.")

        header = ctk.CTkFrame(self.root, corner_radius=18)
        header.pack(fill="x", padx=20, pady=20)
        ctk.CTkLabel(
            header, text="Keystroke Authentication", font=ctk.CTkFont(size=28, weight="bold")
        ).pack(anchor="w", padx=20, pady=(18, 4))
        ctk.CTkLabel(header, textvariable=self.status_var, font=ctk.CTkFont(size=14)).pack(
            anchor="w", padx=20, pady=(0, 18)
        )

        tab_view = ctk.CTkTabview(self.root, corner_radius=18)
        tab_view.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        self.tab_view = tab_view

        calibration = tab_view.add("Stage 1 Calibration")
        feature_stage = tab_view.add("Stage 2 Passphrase")
        enrollment = tab_view.add("Stage 3 Data Collection")
        modeling = tab_view.add("Stage 4 Modeling")
        authentication = tab_view.add("Stage 5 Authentication")
        evaluation = tab_view.add("Stage 6 Evaluation")
        settings = tab_view.add("Settings")

        self._build_calibration_tab(ctk, calibration)
        self._build_feature_tab(ctk, feature_stage)
        self._build_enrollment_tab(ctk, enrollment)
        self._build_modeling_tab(ctk, modeling)
        self._build_auth_tab(ctk, authentication)
        self._build_evaluation_tab(ctk, evaluation)
        self._build_settings_tab(ctk, settings)

        self.root.mainloop()

    def _build_section_card(self, ctk, parent, title: str, description: str):
        card = ctk.CTkScrollableFrame(parent, corner_radius=18)
        card.pack(fill="both", expand=True, padx=16, pady=16)
        ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=20, weight="bold")).pack(
            anchor="w", padx=16, pady=(16, 10)
        )
        ctk.CTkLabel(card, text=description, wraplength=700, justify="left").pack(
            anchor="w", padx=16, pady=(0, 12)
        )
        return card

    def _add_labeled_entry(self, ctk, parent, label: str, variable):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        ctk.CTkLabel(frame, text=label, anchor="w", justify="left").pack(
            anchor="w", padx=0, pady=(0, 4)
        )
        ctk.CTkEntry(frame, textvariable=variable, width=140).pack(anchor="w", padx=0, pady=(0, 0))
        return frame

    def _build_calibration_tab(self, ctk, parent) -> None:
        card = self._build_section_card(
            ctk,
            parent,
            "Stage 1 Calibration",
            "Press the space bar 3 times with a normal force. The app will estimate the fixed key-sound length L and persist it in config.json.",
        )

        self.calibration_sample_rate_var = ctk.StringVar(value=str(self.config.sample_rate))
        param_frame = ctk.CTkFrame(card, fg_color="transparent")
        param_frame.pack(fill="x", padx=16, pady=(0, 12))
        ctk.CTkLabel(param_frame, text="Sample Rate (Hz):", anchor="w").pack(
            side="left", padx=(0, 8)
        )
        ctk.CTkEntry(param_frame, textvariable=self.calibration_sample_rate_var, width=100).pack(
            side="left"
        )

        ctk.CTkButton(card, text="Run Calibration", command=self.run_calibration).pack(
            anchor="w", padx=16, pady=(0, 16)
        )
        self.calibration_notice_var = ctk.StringVar(value="")
        ctk.CTkLabel(
            card,
            textvariable=self.calibration_notice_var,
            wraplength=700,
            justify="left",
            text_color="#7CFF9B",
        ).pack(anchor="w", padx=16, pady=(0, 16))

    def _build_feature_tab(self, ctk, parent) -> None:
        card = self._build_section_card(
            ctk,
            parent,
            "Stage 2 Passphrase",
            "Set the passphrase here first. After saving, the app will jump to Stage 3 automatically.",
        )
        ctk.CTkEntry(card, textvariable=self.passphrase_var, width=460).pack(
            anchor="w", padx=16, pady=(0, 12)
        )
        ctk.CTkButton(
            card, text="Save Passphrase to config.json", command=self.save_passphrase
        ).pack(anchor="w", padx=16, pady=(0, 16))

    def _build_enrollment_tab(self, ctk, parent) -> None:
        card = self._build_section_card(
            ctk,
            parent,
            "Stage 3 Data Collection",
            "Set the number of required samples, then press Start to open the capture window.",
        )
        ctk.CTkLabel(card, text="Required samples:", anchor="w", justify="left").pack(
            anchor="w", padx=16, pady=(0, 4)
        )
        self.collection_target_var = ctk.StringVar(value="100")
        ctk.CTkEntry(card, textvariable=self.collection_target_var, width=140).pack(
            anchor="w", padx=16, pady=(0, 12)
        )
        
        ctk.CTkLabel(card, text="Number of sessions:", anchor="w", justify="left").pack(
            anchor="w", padx=16, pady=(0, 4)
        )
        self.collection_session_var = ctk.StringVar(value=str(getattr(self.config, "collection_session_count", 4)))
        ctk.CTkEntry(card, textvariable=self.collection_session_var, width=140).pack(
            anchor="w", padx=16, pady=(0, 12)
        )

        ctk.CTkButton(
            card,
            text="Start Data Collection",
            command=lambda: self.start_bulk_collection(
                self.config.dataset_path,
                int(self.collection_target_var.get().strip())
                if self.collection_target_var and self.collection_target_var.get().strip().isdigit()
                else 100,
                "Stage 3 Data Collection",
                int(self.collection_session_var.get().strip())
                if hasattr(self, "collection_session_var") and self.collection_session_var.get().strip().isdigit()
                else 4,
            ),
        ).pack(anchor="w", padx=16, pady=(0, 16))
        ctk.CTkLabel(
            card,
            text="The data collection window will clear after each Enter key press and ask for the passphrase again.",
            wraplength=700,
            justify="left",
        ).pack(anchor="w", padx=16, pady=(0, 16))

    def _is_valid_biometric_key(self, event) -> bool:
        invalid_keys = {
            "Return",
            "KP_Enter",
            "Shift_L",
            "Shift_R",
            "Control_L",
            "Control_R",
            "Alt_L",
            "Alt_R",
            "Super_L",
            "Super_R",
            "Caps_Lock",
            "Tab",
            "Escape",
        }
        return event.keysym not in invalid_keys

    def _build_modeling_tab(self, ctk, parent) -> None:
        card = self._build_section_card(
            ctk,
            parent,
            "Stage 4 Modeling",
            "Choose the CSV path, then train the StandardScaler and One-Class SVM with fixed RBF kernel.",
        )
        self.model_dataset_path_var = ctk.StringVar(value=self.config.dataset_path)
        self.model_nu_var = ctk.StringVar(value="0.1")
        self.model_gamma_var = ctk.StringVar(value="scale")
        self.model_algorithm_var = ctk.StringVar(value="lof")
        self.model_contamination_var = ctk.StringVar(value="0.05")
        self.model_n_components_var = ctk.StringVar(value="5")
        self.model_n_estimators_var = ctk.StringVar(value="100")

        ctk.CTkLabel(card, text="CSV path:", anchor="w", justify="left").pack(
            anchor="w", padx=16, pady=(0, 4)
        )
        ctk.CTkEntry(card, textvariable=self.model_dataset_path_var, width=520).pack(
            anchor="w", padx=16, pady=(0, 12)
        )

        ctk.CTkLabel(card, text="Algorithm:", anchor="w", justify="left").pack(
            anchor="w", padx=16, pady=(0, 4)
        )
        ctk.CTkOptionMenu(
            card,
            variable=self.model_algorithm_var,
            values=["lof", "iforest", "svm", "pca_svm"],
            command=self._on_algorithm_change,
        ).pack(anchor="w", padx=16, pady=(0, 12))

        self.model_param_grid = ctk.CTkFrame(card, fg_color="transparent")
        self.model_param_grid.pack(fill="x", padx=16, pady=(0, 12))

        self.model_nu_frame = self._add_labeled_entry(
            ctk, self.model_param_grid, "nu", self.model_nu_var
        )
        self.model_gamma_frame = self._add_labeled_entry(
            ctk, self.model_param_grid, "gamma", self.model_gamma_var
        )
        self.model_contamination_frame = self._add_labeled_entry(
            ctk, self.model_param_grid, "contamination (0.001 - 0.5)", self.model_contamination_var
        )
        self.model_n_components_frame = self._add_labeled_entry(
            ctk, self.model_param_grid, "n_components", self.model_n_components_var
        )
        self.model_n_estimators_frame = self._add_labeled_entry(
            ctk, self.model_param_grid, "n_estimators", self.model_n_estimators_var
        )

        self._on_algorithm_change(self.model_algorithm_var.get())

        ctk.CTkButton(card, text="Train Model From CSV", command=self.train_model).pack(
            anchor="w", padx=16, pady=(0, 16)
        )

    def _on_algorithm_change(self, choice: str) -> None:
        if not hasattr(self, "model_nu_frame"):
            return

        self.model_nu_frame.grid_remove()
        self.model_gamma_frame.grid_remove()
        self.model_contamination_frame.grid_remove()
        self.model_n_components_frame.grid_remove()
        self.model_n_estimators_frame.grid_remove()

        if choice in ("svm", "pca_svm"):
            self.model_nu_frame.grid(row=0, column=0, padx=(0, 16), pady=(0, 12), sticky="w")
            self.model_gamma_frame.grid(row=0, column=1, padx=(0, 16), pady=(0, 12), sticky="w")
            if choice == "pca_svm":
                self.model_n_components_frame.grid(
                    row=0, column=2, padx=(0, 16), pady=(0, 12), sticky="w"
                )
        else:
            self.model_contamination_frame.grid(
                row=0, column=0, padx=(0, 16), pady=(0, 12), sticky="w"
            )
            if choice == "iforest":
                self.model_n_estimators_frame.grid(
                    row=0, column=1, padx=(0, 16), pady=(0, 12), sticky="w"
                )

    def _build_auth_tab(self, ctk, parent) -> None:
        card = self._build_section_card(
            ctk,
            parent,
            "Stage 5 Authentication",
            "Authenticate using the trained model and passphrase.",
        )
        self.auth_model_path_var = ctk.StringVar(value=self.config.model_path)
        self.auth_scaler_path_var = ctk.StringVar(value=self.config.scaler_path)
        self.auth_gmail_var = ctk.StringVar(value=self.config.smtp_recipient)
        self.test_mode_var = ctk.BooleanVar(value=False)
        self.auth_otp_dialog = None
        self.login_success_var = ctk.StringVar(value="")
        self.auth_threshold_var = ctk.StringVar(value=str(self.config.auth_threshold))

        ctk.CTkLabel(card, text="Model path:", anchor="w", justify="left").pack(
            anchor="w", padx=16, pady=(0, 4)
        )
        ctk.CTkEntry(card, textvariable=self.auth_model_path_var, width=520).pack(
            anchor="w", padx=16, pady=(0, 8)
        )

        ctk.CTkLabel(card, text="Scaler path:", anchor="w", justify="left").pack(
            anchor="w", padx=16, pady=(0, 4)
        )
        ctk.CTkEntry(card, textvariable=self.auth_scaler_path_var, width=520).pack(
            anchor="w", padx=16, pady=(0, 12)
        )

        ctk.CTkLabel(card, text="Recipient Gmail (for 2FA):", anchor="w", justify="left").pack(
            anchor="w", padx=16, pady=(0, 4)
        )
        ctk.CTkEntry(card, textvariable=self.auth_gmail_var, width=520).pack(
            anchor="w", padx=16, pady=(0, 12)
        )

        ctk.CTkCheckBox(
            card, text="Test Mode (Disable actual SMTP email sending)", variable=self.test_mode_var
        ).pack(anchor="w", padx=16, pady=(0, 12))

        ctk.CTkLabel(card, text="AUTH_THRESHOLD:", anchor="w", justify="left").pack(
            anchor="w", padx=16, pady=(0, 4)
        )
        ctk.CTkEntry(card, textvariable=self.auth_threshold_var, width=180).pack(
            anchor="w", padx=16, pady=(0, 12)
        )

        ctk.CTkLabel(
            card,
            text="Authenticate opens a passphrase capture window.",
            wraplength=700,
            justify="left",
        ).pack(anchor="w", padx=16, pady=(0, 12))
        button_row = ctk.CTkFrame(card, fg_color="transparent")
        button_row.pack(anchor="w", padx=16, pady=(0, 16), fill="x")
        ctk.CTkButton(button_row, text="Authenticate", command=self.authenticate).pack(side="left")

        self.success_frame = ctk.CTkFrame(card, corner_radius=16)
        self.success_frame.pack(fill="x", padx=16, pady=(0, 16))
        self.success_frame.pack_forget()
        ctk.CTkLabel(
            self.success_frame,
            text="Login success",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(anchor="w", padx=16, pady=(16, 6))
        ctk.CTkLabel(
            self.success_frame,
            textvariable=self.login_success_var,
            wraplength=700,
            justify="left",
        ).pack(anchor="w", padx=16, pady=(0, 16))

    def _build_evaluation_tab(self, ctk, parent) -> None:
        card = self._build_section_card(
            ctk, parent, "Stage 6 Evaluation", "Record test sets and evaluate model FAR/FRR."
        )
        self.eval_owner_path_var = ctk.StringVar(
            value=str(self.config_path.parent / "test_owner.csv")
        )
        self.eval_imposter_path_var = ctk.StringVar(
            value=str(self.config_path.parent / "test_imposter.csv")
        )
        self.eval_model_path_var = ctk.StringVar(value=self.config.model_path)
        self.eval_scaler_path_var = ctk.StringVar(value=self.config.scaler_path)
        self.eval_threshold_var = ctk.StringVar(value=str(self.config.auth_threshold))

        grid = ctk.CTkFrame(card, fg_color="transparent")
        grid.pack(fill="x", padx=16, pady=4)

        self.eval_owner_target_var = ctk.StringVar(value="5")
        self.eval_imposter_target_var = ctk.StringVar(value="5")

        self._add_labeled_entry(ctk, grid, "Owner Test CSV:", self.eval_owner_path_var).grid(
            row=0, column=0, padx=(0, 16), pady=4, sticky="w"
        )
        self._add_labeled_entry(
            ctk, grid, "Owner Target Samples:", self.eval_owner_target_var
        ).grid(row=1, column=0, padx=(0, 16), pady=4, sticky="w")

        self._add_labeled_entry(ctk, grid, "Imposter Test CSV:", self.eval_imposter_path_var).grid(
            row=0, column=1, padx=(0, 16), pady=4, sticky="w"
        )
        self._add_labeled_entry(
            ctk, grid, "Imposter Target Samples:", self.eval_imposter_target_var
        ).grid(row=1, column=1, padx=(0, 16), pady=4, sticky="w")

        btn_grid = ctk.CTkFrame(card, fg_color="transparent")
        btn_grid.pack(fill="x", padx=16, pady=12)

        ctk.CTkButton(
            btn_grid,
            text="Record Owner Sample",
            command=lambda: self.start_bulk_collection(
                self.eval_owner_path_var.get().strip(),
                int(self.eval_owner_target_var.get().strip())
                if self.eval_owner_target_var.get().strip().isdigit()
                else 5,
                "Record Owner Test Samples",
            ),
        ).pack(side="left", padx=(0, 16))

        ctk.CTkButton(
            btn_grid,
            text="Record Imposter Sample",
            command=lambda: self.start_bulk_collection(
                self.eval_imposter_path_var.get().strip(),
                int(self.eval_imposter_target_var.get().strip())
                if self.eval_imposter_target_var.get().strip().isdigit()
                else 5,
                "Record Imposter Test Samples",
            ),
        ).pack(side="left")

        ctk.CTkFrame(card, height=2, fg_color="gray").pack(fill="x", padx=16, pady=12)

        model_grid = ctk.CTkFrame(card, fg_color="transparent")
        model_grid.pack(fill="x", padx=16, pady=4)
        self._add_labeled_entry(ctk, model_grid, "Model Path:", self.eval_model_path_var).grid(
            row=0, column=0, padx=(0, 16), pady=4, sticky="w"
        )
        self._add_labeled_entry(ctk, model_grid, "Scaler Path:", self.eval_scaler_path_var).grid(
            row=0, column=1, padx=(0, 16), pady=4, sticky="w"
        )
        self._add_labeled_entry(ctk, model_grid, "AUTH_THRESHOLD:", self.eval_threshold_var).grid(
            row=1, column=0, padx=(0, 16), pady=4, sticky="w"
        )

        ctk.CTkButton(card, text="Evaluate Model", command=self._evaluate_model).pack(
            anchor="w", padx=16, pady=12
        )

        self.eval_results_var = ctk.StringVar(value="Results will appear here.")
        ctk.CTkLabel(
            card,
            textvariable=self.eval_results_var,
            justify="left",
            font=ctk.CTkFont(family="Courier", size=14),
        ).pack(anchor="w", padx=16, pady=(0, 16))

    def _evaluate_model(self) -> None:
        from keystroke_auth.modeling import (
            load_artifacts,
            load_feature_matrix,
            score_with_artifacts,
        )

        owner_path = self.eval_owner_path_var.get().strip()
        imposter_path = self.eval_imposter_path_var.get().strip()
        model_path = self.eval_model_path_var.get().strip()
        scaler_path = self.eval_scaler_path_var.get().strip()

        try:
            artifacts = load_artifacts(model_path, scaler_path)

            owner_matrix = (
                load_feature_matrix(owner_path) if Path(owner_path).exists() else np.zeros((0, 46))
            )
            imposter_matrix = (
                load_feature_matrix(imposter_path)
                if Path(imposter_path).exists()
                else np.zeros((0, 46))
            )

            if owner_matrix.shape[0] == 0 or imposter_matrix.shape[0] == 0:
                self.eval_results_var.set(
                    "Error: Both owner and imposter test datasets are required for threshold calibration."
                )
                self._set_status("Evaluation failed.")
                return

            owner_scores = [score_with_artifacts(artifacts, row) for row in owner_matrix]
            imposter_scores = [score_with_artifacts(artifacts, row) for row in imposter_matrix]
            threshold, far, frr = calibrate_auth_threshold(owner_scores, imposter_scores)

            self.config.auth_threshold = threshold
            if self.auth_threshold_var is not None:
                self.auth_threshold_var.set(f"{threshold:.6f}")
            if self.eval_threshold_var is not None:
                self.eval_threshold_var.set(f"{threshold:.6f}")
            save_config(self.config, self.config_path)

            owner_preds = [1 if score > threshold else -1 for score in owner_scores]
            imposter_preds = [1 if score > threshold else -1 for score in imposter_scores]

            total_samples = len(owner_preds) + len(imposter_preds)
            correct = sum(1 for p in owner_preds if p == 1) + sum(
                1 for p in imposter_preds if p == -1
            )
            accuracy = correct / total_samples if total_samples > 0 else 0.0

            results = (
                f"--- Evaluation Results ---\n"
                f"Model: {Path(model_path).name}\n"
                f"Owner Samples (True Positives targeted): {len(owner_preds)}\n"
                f"Imposter Samples (True Negatives targeted): {len(imposter_preds)}\n\n"
                f"AUTH_THRESHOLD: {threshold:.6f}\n"
                f"FAR (False Acceptance Rate): {far * 100:.2f}%\n"
                f"FRR (False Rejection Rate): {frr * 100:.2f}%\n"
                f"Overall Accuracy: {accuracy * 100:.2f}%\n"
            )
            self.eval_results_var.set(results)
            self._set_status("Evaluation complete.")
        except Exception as exc:
            self.eval_results_var.set(f"Evaluation Failed: {exc}")
            self._set_status(f"Evaluation Failed: {exc}")

    def _build_settings_tab(self, ctk, parent) -> None:
        card = self._build_section_card(
            ctk, parent, "SMTP Configuration", "Configure SMTP sender credentials for 2FA."
        )

        self.smtp_host_var = ctk.StringVar(value=self.config.smtp_host)
        self.smtp_port_var = ctk.StringVar(value=str(self.config.smtp_port))
        self.smtp_username_var = ctk.StringVar(value=self.config.smtp_username)
        self.smtp_password_var = ctk.StringVar(value=self.config.smtp_password)
        self.collection_session_count_var = ctk.StringVar(
            value=str(self.config.collection_session_count)
        )
        self.white_noise_std_var = ctk.StringVar(value=str(self.config.white_noise_std))
        self.settings_threshold_var = ctk.StringVar(value=str(self.config.auth_threshold))

        fields = [
            ("SMTP Host:", self.smtp_host_var, False),
            ("SMTP Port:", self.smtp_port_var, False),
            ("SMTP Sender Username:", self.smtp_username_var, False),
            ("SMTP Sender Password (App Password):", self.smtp_password_var, True),
        ]

        for label_text, str_var, is_password in fields:
            ctk.CTkLabel(card, text=label_text, anchor="w", justify="left").pack(
                anchor="w", padx=16, pady=(0, 4)
            )
            entry = ctk.CTkEntry(card, textvariable=str_var, width=520)
            if is_password:
                entry.configure(show="*")
            entry.pack(anchor="w", padx=16, pady=(0, 8))

        ctk.CTkFrame(card, height=2, fg_color="gray").pack(fill="x", padx=16, pady=(8, 12))

        ctk.CTkLabel(card, text="Collection Session Count:", anchor="w", justify="left").pack(
            anchor="w", padx=16, pady=(0, 4)
        )
        ctk.CTkEntry(card, textvariable=self.collection_session_count_var, width=180).pack(
            anchor="w", padx=16, pady=(0, 8)
        )

        ctk.CTkLabel(card, text="White Noise Std:", anchor="w", justify="left").pack(
            anchor="w", padx=16, pady=(0, 4)
        )
        ctk.CTkEntry(card, textvariable=self.white_noise_std_var, width=180).pack(
            anchor="w", padx=16, pady=(0, 8)
        )

        ctk.CTkLabel(card, text="AUTH_THRESHOLD:", anchor="w", justify="left").pack(
            anchor="w", padx=16, pady=(0, 4)
        )
        ctk.CTkEntry(card, textvariable=self.settings_threshold_var, width=180).pack(
            anchor="w", padx=16, pady=(0, 12)
        )

        def save_settings() -> None:
            self.config.smtp_host = self.smtp_host_var.get().strip()
            try:
                self.config.smtp_port = int(self.smtp_port_var.get().strip())
            except ValueError:
                self._set_status("SMTP port must be a number.")
                return
            self.config.smtp_username = self.smtp_username_var.get().strip()
            self.config.smtp_password = self.smtp_password_var.get().strip()

            try:
                self.config.collection_session_count = int(
                    self.collection_session_count_var.get().strip()
                )
            except ValueError:
                self._set_status("Collection session count must be a number.")
                return

            try:
                self.config.white_noise_std = float(self.white_noise_std_var.get().strip())
            except ValueError:
                self._set_status("White noise std must be a number.")
                return

            try:
                self.config.auth_threshold = float(self.settings_threshold_var.get().strip())
            except ValueError:
                self._set_status("AUTH_THRESHOLD must be a number.")
                return

            if self.auth_threshold_var is not None:
                self.auth_threshold_var.set(f"{self.config.auth_threshold:.6f}")
            if self.eval_threshold_var is not None:
                self.eval_threshold_var.set(f"{self.config.auth_threshold:.6f}")

            from keystroke_auth.config import save_config

            save_config(self.config)
            self._set_status("Settings saved to config.json.")

        ctk.CTkButton(card, text="Save Settings", command=save_settings).pack(
            anchor="w", padx=16, pady=(8, 16)
        )

    def _switch_to_feature_tab(self) -> None:
        if self.tab_view is not None:
            self.tab_view.set("Stage 3 Data Collection")

    def _switch_to_passphrase_tab(self) -> None:
        if self.tab_view is not None:
            self.tab_view.set("Stage 2 Passphrase")

    def _switch_to_modeling_tab(self) -> None:
        if self.tab_view is not None:
            self.tab_view.set("Stage 4 Modeling")

    def _switch_to_auth_tab(self) -> None:
        if self.tab_view is not None:
            self.tab_view.set("Stage 5 Authentication")

    def _open_otp_dialog(self) -> None:
        if self.auth_otp_dialog is not None:
            return
        ctk = self._require_ctk()
        self.auth_otp_dialog = ctk.CTkToplevel(self.root)
        self.auth_otp_dialog.title("OTP Challenge")
        self.auth_otp_dialog.geometry("400x200")
        self.auth_otp_dialog.grab_set()

        frame = ctk.CTkFrame(self.auth_otp_dialog, corner_radius=18)
        frame.pack(fill="both", expand=True, padx=16, pady=16)
        ctk.CTkLabel(frame, text="OTP challenge", font=ctk.CTkFont(size=20, weight="bold")).pack(
            anchor="w", padx=16, pady=(16, 8)
        )
        ctk.CTkLabel(
            frame,
            text="Enter the 6-digit OTP code sent to your email (or printed in terminal) and press Enter.",
            wraplength=340,
            justify="left",
        ).pack(anchor="w", padx=16, pady=(0, 8))

        self.auth_otp_entry = ctk.CTkEntry(frame, placeholder_text="OTP code", width=340)
        self.auth_otp_entry.pack(anchor="w", padx=16, pady=(0, 16))
        self.auth_otp_entry.bind("<Return>", lambda _event: self.verify_otp())
        self.auth_otp_dialog.after_idle(self.auth_otp_entry.focus_set)

    def _hide_otp_interface(self) -> None:
        if self.auth_otp_dialog is not None:
            self.auth_otp_dialog.destroy()
            self.auth_otp_dialog = None

    def _show_otp_interface(self) -> None:
        if self.otp_frame is not None:
            self.otp_frame.pack_forget()
            self.otp_frame.pack(fill="x", padx=16, pady=(0, 16))
        if self.auth_otp_entry is not None:
            self.auth_otp_entry.focus_set()

    def _hide_login_success(self) -> None:
        pass

    def _show_login_success(self, message: str) -> None:
        ctk = self._require_ctk()
        success_dialog = ctk.CTkToplevel(self.root)
        success_dialog.title("Access Granted")
        success_dialog.geometry("500x250")
        success_dialog.grab_set()

        frame = ctk.CTkFrame(success_dialog, corner_radius=18, fg_color="#2ECC71")
        frame.pack(fill="both", expand=True, padx=16, pady=16)

        ctk.CTkLabel(
            frame, text="SUCCESS", font=ctk.CTkFont(size=48, weight="bold"), text_color="white"
        ).pack(expand=True, pady=(20, 0))
        ctk.CTkLabel(
            frame,
            text=message,
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="white",
            wraplength=400,
        ).pack(pady=(10, 20))

        ctk.CTkButton(
            frame,
            text="OK",
            command=success_dialog.destroy,
            fg_color="white",
            text_color="#2ECC71",
            font=ctk.CTkFont(weight="bold"),
        ).pack(pady=(0, 20))

    def _refresh_login_success_visibility(self) -> None:
        if self.success_frame is not None:
            self.success_frame.pack_forget()

    def save_passphrase(self) -> None:
        passphrase = self.passphrase_var.get().strip()
        if not passphrase:
            self._set_status("Passphrase cannot be empty.")
            return

        self.config.passphrase = passphrase
        save_config(self.config, self.config_path)
        self._switch_to_feature_tab()
        self._set_status("Passphrase saved to config.json. Please continue in Stage 3.")

    def start_bulk_collection(
        self, target_filepath: str, target_count: int, title: str = "Data Collection", session_count: int | None = None
    ) -> None:
        if not self.config.passphrase:
            self._set_status("Please set a passphrase in Stage 2 first.")
            return

        if target_count <= 0:
            self._set_status("Required sample count must be greater than zero.")
            return

        self.collection_active = True
        self.collection_expected_count = target_count
        self.collection_current_index = 0
        self.collection_target_filepath = target_filepath
        
        if session_count is None:
            session_count = max(
                1, min(target_count, int(getattr(self.config, "collection_session_count", 4)))
            )
        else:
            session_count = max(1, min(target_count, session_count))
            
        self.collection_session_count = session_count
        self.collection_session_plan = self._build_collection_session_plan(
            target_count, self.collection_session_count
        )
        self.collection_session_index = 0
        self.collection_session_sample_count = 0
        self.collection_session_paused = False
        self._open_bulk_dialog(title)
        self._begin_bulk_sample()

    def _open_bulk_dialog(self, title: str) -> None:
        ctk = self._require_ctk()
        self.collection_dialog = ctk.CTkToplevel(self.root)
        self.collection_dialog.title(title)
        self.collection_dialog.geometry("760x420")
        self.collection_dialog.protocol("WM_DELETE_WINDOW", self._cancel_bulk_collection)

        container = ctk.CTkFrame(self.collection_dialog, corner_radius=18)
        container.pack(fill="both", expand=True, padx=16, pady=16)
        ctk.CTkLabel(
            container,
            text=title,
            font=ctk.CTkFont(size=22, weight="bold"),
        ).pack(anchor="w", padx=16, pady=(16, 8))
        self.collection_progress_var = ctk.StringVar(value="0 / 0")
        self.collection_prompt_var = ctk.StringVar(value="")
        ctk.CTkLabel(container, textvariable=self.collection_progress_var).pack(
            anchor="w", padx=16, pady=(0, 8)
        )
        ctk.CTkLabel(
            container,
            text=f"Passphrase: {self.config.passphrase}",
            wraplength=700,
            justify="left",
        ).pack(anchor="w", padx=16, pady=(0, 8))
        ctk.CTkLabel(
            container,
            textvariable=self.collection_prompt_var,
            wraplength=700,
            justify="left",
        ).pack(anchor="w", padx=16, pady=(0, 12))
        self.collection_input_var = ctk.StringVar(value="")
        self.collection_input_entry = ctk.CTkEntry(
            container, textvariable=self.collection_input_var, width=420
        )
        self.collection_input_entry.pack(anchor="w", padx=16, pady=(0, 12))
        self.collection_continue_button = ctk.CTkButton(
            container, text="Continue Next Session", command=self._resume_bulk_session
        )
        self.collection_dialog.after_idle(self.collection_input_entry.focus_set)
        ctk.CTkButton(container, text="Cancel", command=self._cancel_bulk_collection).pack(
            anchor="w", padx=16, pady=(0, 16)
        )

    def _build_collection_session_plan(self, total_count: int, session_count: int) -> list[int]:
        total_count = max(int(total_count), 0)
        session_count = max(int(session_count), 1)
        if total_count == 0:
            return []
        session_count = min(session_count, total_count)
        base = total_count // session_count
        remainder = total_count % session_count
        return [base + (1 if index < remainder else 0) for index in range(session_count)]

    def _cancel_bulk_collection(self) -> None:
        self.collection_active = False
        self.collection_session_paused = False
        if self.collection_capture_stop_event is not None:
            self.collection_capture_stop_event.set()
        if self.collection_capture_thread is not None:
            self.collection_capture_thread.join()
        if self.collection_dialog is not None:
            self.collection_dialog.unbind_all("<KeyRelease>")
            self.collection_dialog.unbind_all("<Return>")
            self.collection_dialog.unbind_all("<KP_Enter>")
            self.collection_dialog.destroy()
        self.collection_dialog = None
        self.collection_capture_thread = None
        self.collection_capture_stop_event = None
        self.collection_capture_audio = None
        self.collection_input_entry = None
        self.collection_input_var = None
        self.collection_continue_button = None
        self._set_status("Data collection canceled.")

    def _begin_bulk_sample(self) -> None:
        if not self.collection_active:
            return

        if self.collection_current_index >= self.collection_expected_count:
            self._finish_bulk_collection()
            return

        self.collection_capture_audio = None
        self.collection_capture_timestamps = []
        self.collection_capture_keysyms = []
        self.collection_capture_previous_length = 0
        self.collection_capture_started_at = perf_counter()
        self.collection_expected_text = self.config.passphrase.strip()

        if self.collection_progress_var is not None:
            self.collection_progress_var.set(
                f"{self.collection_current_index} / {self.collection_expected_count} collected"
            )
        if self.collection_prompt_var is not None:
            session_total = (
                self.collection_session_plan[self.collection_session_index]
                if self.collection_session_plan
                and self.collection_session_index < len(self.collection_session_plan)
                else self.collection_expected_count
            )
            self.collection_prompt_var.set(
                f"Session {self.collection_session_index + 1}/{len(self.collection_session_plan) or 1} | "
                f"Sample {self.collection_session_sample_count + 1} of {session_total}: type the passphrase and press Enter to submit."
            )
        if self.collection_input_var is not None:
            self.collection_input_var.set("")
        if self.collection_input_entry is not None:
            self.collection_dialog.after_idle(self.collection_input_entry.focus_set)
        if self.collection_continue_button is not None:
            self.collection_continue_button.pack_forget()

        from threading import Event

        self.collection_capture_stop_event = Event()

        def audio_task() -> None:
            from .capture import AudioRecorder

            recorder = AudioRecorder(sample_rate=self.config.sample_rate)

            def set_start():
                self.collection_capture_started_at = perf_counter()

            if self.collection_capture_stop_event is not None:
                self.collection_capture_audio = recorder.record_until(
                    self.collection_capture_stop_event, on_start=set_start
                )
            else:
                self.collection_capture_audio = recorder.record_until(Event(), on_start=set_start)

        self.collection_capture_thread = Thread(target=audio_task, daemon=True)
        self.collection_capture_thread.start()

        if self.collection_dialog is not None:
            self.collection_dialog.unbind_all("<KeyPress>")
            self.collection_dialog.unbind_all("<KeyRelease>")
            self.collection_dialog.bind_all("<KeyPress>", self._on_bulk_collection_key_press)
            self.collection_dialog.bind_all("<KeyRelease>", self._on_bulk_collection_key_release)
            self.collection_dialog.bind_all("<Return>", self._on_bulk_collection_submit, add="+")
            self.collection_dialog.bind_all("<KP_Enter>", self._on_bulk_collection_submit, add="+")

    def _finish_bulk_collection(self) -> None:
        self.collection_active = False
        self.collection_session_paused = False
        if self.collection_capture_stop_event is not None:
            self.collection_capture_stop_event.set()
        if self.collection_capture_thread is not None:
            self.collection_capture_thread.join()
        if self.collection_dialog is not None:
            self.collection_dialog.unbind_all("<KeyPress>")
            self.collection_dialog.unbind_all("<KeyRelease>")
            self.collection_dialog.unbind_all("<Return>")
            self.collection_dialog.unbind_all("<KP_Enter>")
            self.collection_dialog.destroy()

        self.collection_dialog = None
        self.collection_capture_thread = None
        self.collection_capture_stop_event = None
        self.collection_capture_audio = None
        self.collection_input_entry = None
        self.collection_input_var = None
        self.collection_continue_button = None

        self._set_status("Bulk collection complete. CSV has been written.")

    def _pause_bulk_session(self) -> None:
        self.collection_session_paused = True
        if self.collection_capture_stop_event is not None:
            self.collection_capture_stop_event.set()
        if self.collection_capture_thread is not None:
            self.collection_capture_thread.join()

        if self.collection_dialog is not None:
            self.collection_dialog.unbind_all("<KeyPress>")
            self.collection_dialog.unbind_all("<KeyRelease>")
            self.collection_dialog.unbind_all("<Return>")
            self.collection_dialog.unbind_all("<KP_Enter>")

        if self.collection_continue_button is not None:
            self.collection_continue_button.pack(anchor="w", padx=16, pady=(0, 12))

        if self.collection_prompt_var is not None:
            self.collection_prompt_var.set(
                "Session pause. Change the laptop angle or surrounding noise, then click Continue Next Session to resume."
            )
        if self.collection_input_entry is not None:
            self.collection_input_entry.configure(state="disabled")

    def _resume_bulk_session(self) -> None:
        if not self.collection_session_paused:
            return

        self.collection_session_paused = False
        if self.collection_input_entry is not None:
            self.collection_input_entry.configure(state="normal")
        if self.collection_continue_button is not None:
            self.collection_continue_button.pack_forget()

        self.collection_session_index += 1
        self.collection_session_sample_count = 0
        self._begin_bulk_sample()

    def _augment_training_audio(self, audio: np.ndarray) -> np.ndarray:
        noise_std = float(getattr(self.config, "white_noise_std", self.collection_noise_std))
        if audio.size == 0 or noise_std <= 0.0:
            return audio

        rng = np.random.default_rng()
        if rng.random() > 0.35:
            return audio

        scaled_std = noise_std * float(rng.uniform(0.25, 1.0))
        noisy_audio = audio + rng.normal(0.0, scaled_std, size=audio.shape)
        return np.clip(noisy_audio, -1.0, 1.0)

    def _restart_bulk_sample(self, message: str) -> None:
        if self.collection_prompt_var is not None:
            self.collection_prompt_var.set(message)
        if self.collection_input_var is not None:
            self.collection_input_var.set("")
        if self.collection_input_entry is not None:
            self.collection_input_entry.configure(state="disabled")

        if self.collection_capture_stop_event is not None:
            self.collection_capture_stop_event.set()
        if self.collection_capture_thread is not None:
            self.collection_capture_thread.join()

        def resume():
            if self.collection_input_entry is not None:
                self.collection_input_entry.configure(state="normal")
            self._begin_bulk_sample()

        self.root.after(500, resume)

    def _on_bulk_collection_submit(self, event) -> str:
        if not self.collection_active or self.collection_input_var is None:
            return "break"

        current_text = self.collection_input_var.get().strip()
        if current_text != self.collection_expected_text:
            self._restart_bulk_sample("Input error. Please try again.")
            return "break"

        self._complete_bulk_sample()
        return "break"

    def _on_bulk_collection_key_press(self, event) -> None:
        if not self.collection_active or self.collection_input_var is None:
            return

        if self._is_valid_biometric_key(event):
            self.collection_capture_timestamps.append(perf_counter())
            self.collection_capture_keysyms.append(event.keysym)

    def _on_bulk_collection_key_release(self, event) -> None:
        pass

    def _complete_bulk_sample(self) -> None:
        if self.collection_capture_stop_event is not None:
            self.collection_capture_stop_event.set()
        if self.collection_capture_thread is not None:
            self.collection_capture_thread.join()

        audio = self.collection_capture_audio
        start_time = getattr(self, "collection_capture_started_at", 0.0)
        timestamps = [t - start_time for t in self.collection_capture_timestamps]
        if audio is None:
            self._restart_bulk_sample("No audio captured. Please try again.")
            return

        target_path = getattr(self, "collection_target_filepath", self.config.dataset_path)
        try:
            if Path(target_path).resolve() == Path(self.config.dataset_path).resolve():
                audio = self._augment_training_audio(audio)
        except Exception:
            pass

        try:
            features = extract_46_features(
                audio,
                timestamps,
                self.config.sample_rate,
                self.config.key_length,
                self.config.histogram_bins,
                keysyms=self.collection_capture_keysyms,
            )
            append_feature_row(target_path, features)
        except Exception as exc:
            self._restart_bulk_sample(f"Sample failed: {exc}")
            return

        self.collection_session_sample_count += 1
        self.collection_current_index += 1
        session_target = (
            self.collection_session_plan[self.collection_session_index]
            if self.collection_session_plan
            and self.collection_session_index < len(self.collection_session_plan)
            else self.collection_expected_count
        )
        if (
            self.collection_session_sample_count >= session_target
            and self.collection_current_index < self.collection_expected_count
        ):
            self._pause_bulk_session()
            return

        if self.collection_current_index >= self.collection_expected_count:
            self._finish_bulk_collection()
        else:
            self._begin_bulk_sample()

    def _start_text_capture(
        self,
        expected_text: str,
        mode: str,
        on_success=None,
        on_error=None,
        initial_status: str | None = None,
    ) -> None:
        if self.capture_active:
            self._set_status("A capture session is already running.")
            return

        if self.capture_input_entry is None or self.capture_input_var is None:
            self._set_status("Capture input is not ready.")
            return

        self.capture_active = True
        self.capture_timestamps = []
        self.capture_expected_text = expected_text.strip()
        self.capture_timestamps = []
        self.capture_keysyms = []
        self.capture_started_at = perf_counter()
        self.capture_previous_length = 0
        self.capture_success_callback = on_success
        self.capture_error_callback = on_error
        self.capture_input_var.set("")
        self.capture_input_entry.focus_set()
        if initial_status is not None:
            self._set_status(initial_status)
        else:
            self._set_status(
                f"{mode.capitalize()} capture started. Type the passphrase into the capture box."
            )

        try:
            from .capture import _import_pyaudio  # type: ignore

            _import_pyaudio()
        except DependencyError as exc:
            self.capture_active = False
            self._set_status(f"Stage 2 needs microphone permission and pyaudio. Detail: {exc}")
            return

        from threading import Event

        self.capture_audio_stop_event = Event()

        def audio_task() -> None:
            from .capture import AudioRecorder

            def set_start():
                self.capture_started_at = perf_counter()

            recorder = AudioRecorder(sample_rate=self.config.sample_rate)
            if self.capture_audio_stop_event is not None:
                audio = recorder.record_until(self.capture_audio_stop_event, on_start=set_start)
            else:
                audio = recorder.record_until(Event(), on_start=set_start)
            self.capture_audio = audio

        self.capture_audio = None
        self.capture_audio_thread = Thread(target=audio_task, daemon=True)
        self.capture_audio_thread.start()

        self.root.bind_all("<KeyPress>", self._on_capture_key_press, add="+")
        self.root.bind_all("<KeyRelease>", self._on_capture_key_release, add="+")

    def _finalize_text_capture(self) -> None:
        if not self.capture_active:
            return

        self.capture_active = False
        self.root.unbind_all("<KeyPress>")
        self.root.unbind_all("<KeyRelease>")

        if self.capture_audio_stop_event is not None:
            self.capture_audio_stop_event.set()
        if self.capture_audio_thread is not None:
            self.capture_audio_thread.join()

        audio = self.capture_audio
        start_time = getattr(self, "capture_started_at", 0.0)
        timestamps = [t - start_time for t in self.capture_timestamps]
        expected_text = self.capture_expected_text
        captured_text = (
            self.capture_input_var.get().strip() if self.capture_input_var is not None else ""
        )
        success_callback = self.capture_success_callback
        error_callback = self.capture_error_callback

        self.capture_success_callback = None
        self.capture_error_callback = None

        if audio is None:
            if error_callback:
                error_callback(Exception("Capture audio was not recorded."))
            else:
                self._set_status("Capture audio was not recorded.")
            return

        self._set_status("判定中... (Evaluating...)")
        if self.capture_input_entry is not None:
            self.capture_input_entry.configure(state="disabled")
        self.root.update_idletasks()

        def task():
            features = extract_46_features(
                audio,
                timestamps,
                self.config.sample_rate,
                self.config.key_length,
                self.config.histogram_bins,
                keysyms=self.capture_keysyms,
            )
            return features

        def success(features):
            if success_callback is not None:
                success_callback(features)

        def error(exc):
            if error_callback is not None:
                error_callback(exc)
            else:
                self._set_status(str(exc))

        self._run_background(task, on_success=success, on_error=error)

    def _handle_capture_failure(self, message: str) -> None:
        self.capture_active = False
        self.root.unbind_all("<KeyPress>")
        self.root.unbind_all("<KeyRelease>")
        if self.capture_input_entry is not None:
            self.capture_input_entry.configure(state="disabled")
        if self.capture_audio_stop_event is not None:
            self.capture_audio_stop_event.set()
        self._set_status(message)

        def resume():
            if self.capture_input_entry is not None:
                self.capture_input_entry.configure(state="normal")
            if self.capture_mode == "login":
                self._start_text_capture(
                    self.config.passphrase,
                    mode="login",
                    on_success=lambda features: self._handle_login_features(features),
                    on_error=self.capture_error_callback
                    if hasattr(self, "capture_error_callback") and self.capture_error_callback
                    else (lambda exc: self._set_status(str(exc))),
                    initial_status="Ready. Type your passphrase.",
                )

        self.root.after(500, resume)

    def _on_capture_key_press(self, event) -> None:
        if not self.capture_active or self.capture_input_var is None:
            return

        if event.widget != self.capture_input_entry and event.widget != getattr(self.capture_input_entry, "_entry", None):
            return

        if self._is_valid_biometric_key(event):
            self.capture_timestamps.append(perf_counter())
            self.capture_keysyms.append(event.keysym)

    def _on_capture_key_release(self, event) -> None:
        if not self.capture_active or self.capture_input_var is None:
            return

        if event.widget != self.capture_input_entry and event.widget != getattr(self.capture_input_entry, "_entry", None):
            return

        current_text = self.capture_input_var.get()
        if current_text == self.capture_expected_text:
            self._set_status("Text matches expected passphrase. Press Enter to submit.")

    def run_calibration(self) -> None:
        if self.calibration_active:
            self._set_status("Calibration is already running.")
            return

        try:
            sample_rate = int(self.calibration_sample_rate_var.get().strip())
            self.config.sample_rate = sample_rate
        except ValueError:
            self._set_status("Sample rate must be a valid integer (e.g., 44100 or 48000).")
            return

        self._set_status("Calibration started. Focus the app and press space 3 times.")

        try:
            from .capture import _import_pyaudio  # type: ignore

            _import_pyaudio()
        except DependencyError as exc:
            self._set_status(f"Stage 1 needs microphone permission and pyaudio. Detail: {exc}")
            return

        from threading import Event

        self.calibration_active = True
        self.calibration_press_times = []
        self.calibration_stop_event = Event()

        started = perf_counter()

        def audio_task() -> None:
            from .capture import AudioRecorder  # local import to keep startup light

            recorder = AudioRecorder(sample_rate=self.config.sample_rate)
            if self.calibration_stop_event is not None:
                audio = recorder.record_until(self.calibration_stop_event)
            else:
                audio = recorder.record_until(Event())
            self.calibration_audio = audio

        self.calibration_audio = None
        self.calibration_thread = Thread(target=audio_task, daemon=True)
        self.calibration_thread.start()

        def on_space(event) -> str | None:
            if not self.calibration_active or event.keysym != "space":
                return None

            self.calibration_press_times.append(perf_counter() - started)
            count = len(self.calibration_press_times)
            self._set_status(f"Calibration press {count}/{self.calibration_press_target}")

            if count >= self.calibration_press_target:
                self._complete_calibration()
            return "break"

        self.root.bind("<KeyPress-space>", on_space)
        self.root.focus_force()

    def _complete_calibration(self) -> None:
        if not self.calibration_active:
            return

        self.calibration_active = False
        self.root.unbind("<KeyPress-space>")

        def finalize() -> None:
            if self.calibration_stop_event is not None:
                self.calibration_stop_event.set()
            if self.calibration_thread is not None:
                self.calibration_thread.join()

            audio = getattr(self, "calibration_audio", None)
            if audio is None:
                raise ValueError("Calibration audio was not captured")

            calibration = estimate_key_length(
                audio,
                self.config.sample_rate,
                self.calibration_press_times,
            )
            persist_calibration(self.config, calibration, self.config_path)
            return calibration

        try:
            finalize()
        except Exception as exc:
            self._set_status(f"Calibration failed: {exc}")
            return

        self._switch_to_passphrase_tab()
        message = (
            "Calibration complete. config.json has been written. "
            "Please go to Stage 2 Passphrase next."
        )
        if self.calibration_notice_var is not None:
            self.calibration_notice_var.set(message)
        self._set_status(message)

    def _set_feature_output(self, text: str) -> None:
        self.feature_preview_var.set(text)
        if self.feature_output is not None:
            self.feature_output.configure(state="normal")
            self.feature_output.delete("1.0", "end")
            self.feature_output.insert("end", text)
            self.feature_output.configure(state="disabled")

    def preview_features(self) -> None:
        self.save_passphrase()

    def capture_enrollment(self) -> None:
        passphrase = self.passphrase_var.get().strip()
        if not passphrase:
            self._set_status("Passphrase cannot be empty.")
            return

        self._start_text_capture(
            passphrase,
            mode="enrollment",
            on_success=lambda features: (
                append_feature_row(self.config.dataset_path, features),
                self._set_status("Data row captured and appended to dataset.csv."),
            ),
            on_error=lambda exc: self._set_status(f"Stage 3 data collection failed: {exc}"),
        )

    def train_model(self) -> None:
        dataset_text = (
            self.model_dataset_path_var.get().strip() if self.model_dataset_path_var else ""
        )
        dataset_path = Path(dataset_text or self.config.dataset_path)

        try:
            algorithm_value = (
                self.model_algorithm_var.get().strip()
                if hasattr(self, "model_algorithm_var")
                else "lof"
            )
            if algorithm_value in ("lof", "iforest"):
                nu_value = (
                    float(self.model_contamination_var.get().strip())
                    if hasattr(self, "model_contamination_var")
                    else 0.05
                )
            else:
                nu_value = float(self.model_nu_var.get().strip()) if self.model_nu_var else 0.1

            gamma_text = self.model_gamma_var.get().strip() if self.model_gamma_var else "scale"
            gamma_value: str | float
            if gamma_text.lower() in {"scale", "auto"}:
                gamma_value = gamma_text.lower()
            else:
                gamma_value = float(gamma_text)
        except ValueError as exc:
            self._set_status(f"Stage 4 parameter error: {exc}")
            return

        try:
            n_estimators_value = (
                int(self.model_n_estimators_var.get().strip())
                if hasattr(self, "model_n_estimators_var")
                else 100
            )
            n_components_value = (
                int(self.model_n_components_var.get().strip())
                if hasattr(self, "model_n_components_var")
                else 5
            )
        except ValueError:
            n_estimators_value = 100
            n_components_value = 5

        self._set_status(f"Training model from {dataset_path} ...")

        def task():
            matrix = load_feature_matrix(dataset_path)
            artifacts = train_one_class_model(
                matrix,
                algorithm=algorithm_value,
                nu=nu_value,
                kernel="rbf",
                gamma=gamma_value,
                n_estimators=n_estimators_value,
                n_components=n_components_value,
            )
            save_artifacts(artifacts, self.config.model_path, self.config.scaler_path)
            return artifacts

        def success(_artifacts):
            self.model_loaded = _artifacts
            self._switch_to_auth_tab()
            self._set_status("Model trained and saved. Please continue in Stage 5.")

        self._run_background(task, on_success=success)

    def load_model(self) -> None:
        try:
            from keystroke_auth.modeling import load_artifacts

            model_path = (
                self.auth_model_path_var.get().strip()
                if hasattr(self, "auth_model_path_var")
                else self.config.model_path
            )
            scaler_path = (
                self.auth_scaler_path_var.get().strip()
                if hasattr(self, "auth_scaler_path_var")
                else self.config.scaler_path
            )

            self.model_loaded = load_artifacts(model_path, scaler_path)
            self._set_status("Model loaded successfully.")
        except Exception as exc:
            self._set_status(f"Failed to load model: {exc}")
            self.model_loaded = None

    def _open_auth_gmail_dialog(self) -> None:
        ctk = self._require_ctk()
        self.auth_gmail_dialog = ctk.CTkToplevel(self.root)
        self.auth_gmail_dialog.title("Enter Gmail")
        self.auth_gmail_dialog.geometry("520x220")
        self.auth_gmail_dialog.grab_set()

        frame = ctk.CTkFrame(self.auth_gmail_dialog, corner_radius=18)
        frame.pack(fill="both", expand=True, padx=16, pady=16)
        ctk.CTkLabel(frame, text="Enter Gmail", font=ctk.CTkFont(size=20, weight="bold")).pack(
            anchor="w", padx=16, pady=(16, 8)
        )
        self.auth_gmail_var = ctk.StringVar(value=self.email_var.get())
        gmail_entry = ctk.CTkEntry(frame, textvariable=self.auth_gmail_var, width=360)
        gmail_entry.pack(anchor="w", padx=16, pady=(0, 12), fill="x")

        def confirm_gmail() -> None:
            gmail_value = self.auth_gmail_var.get().strip()
            if not gmail_value:
                self._set_status("Gmail cannot be empty.")
                return
            self.email_var.set(gmail_value)
            if self.auth_gmail_dialog is not None:
                self.auth_gmail_dialog.destroy()
            self.auth_gmail_dialog = None
            self._open_auth_passphrase_dialog()

        ctk.CTkButton(frame, text="Continue", command=confirm_gmail).pack(
            anchor="w", padx=16, pady=(0, 16)
        )
        gmail_entry.bind("<Return>", lambda _event: confirm_gmail())
        self.auth_gmail_dialog.after_idle(gmail_entry.focus_set)

    def _open_auth_passphrase_dialog(self) -> None:
        ctk = self._require_ctk()
        self.auth_passphrase_dialog = ctk.CTkToplevel(self.root)
        self.auth_passphrase_dialog.title("Authenticate")
        self.auth_passphrase_dialog.geometry("760x280")
        self.auth_passphrase_dialog.grab_set()

        frame = ctk.CTkFrame(self.auth_passphrase_dialog, corner_radius=18)
        frame.pack(fill="both", expand=True, padx=16, pady=16)
        ctk.CTkLabel(
            frame,
            text="Type your passphrase to authenticate.",
            font=ctk.CTkFont(size=20, weight="bold"),
        ).pack(anchor="w", padx=16, pady=(16, 8))
        ctk.CTkLabel(
            frame,
            text="The model starts capturing as soon as you begin typing. Press Enter when finished.",
            wraplength=700,
            justify="left",
        ).pack(anchor="w", padx=16, pady=(0, 8))

        self.auth_dialog_status_var = ctk.StringVar(value="")
        ctk.CTkLabel(
            frame,
            textvariable=self.auth_dialog_status_var,
            text_color="red",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).pack(anchor="w", padx=16, pady=(0, 8))

        self.auth_passphrase_var = ctk.StringVar(value="")
        auth_entry = ctk.CTkEntry(frame, textvariable=self.auth_passphrase_var, width=460)
        auth_entry.pack(anchor="w", padx=16, pady=(0, 12))
        self.capture_input_var = self.auth_passphrase_var
        self.capture_input_entry = auth_entry

        def on_error(exc):
            self._set_status(f"Error: {exc}")

        self._start_text_capture(
            self.config.passphrase,
            mode="login",
            on_success=lambda features: self._handle_login_features(features),
            on_error=on_error,
        )
        auth_entry.bind("<Return>", self._on_auth_passphrase_submit)
        auth_entry.bind("<KP_Enter>", self._on_auth_passphrase_submit)
        self.auth_passphrase_dialog.after_idle(auth_entry.focus_set)

    def _close_auth_dialogs(self) -> None:
        if self.auth_passphrase_dialog is not None:
            self.auth_passphrase_dialog.destroy()
        if self.auth_gmail_dialog is not None:
            self.auth_gmail_dialog.destroy()
        self.auth_passphrase_dialog = None
        self.auth_gmail_dialog = None

    def _on_auth_passphrase_submit(self, event) -> str:
        if self.capture_input_var is None:
            return "break"

        captured_text = self.capture_input_var.get().strip()
        if not self.capture_active or captured_text != self.config.passphrase.strip():
            if hasattr(self, "auth_dialog_status_var"):
                self.auth_dialog_status_var.set(
                    "Input Error: Passphrase incorrect or typing error detected. Please re-input."
                )
            self.capture_active = False
            self.root.unbind_all("<KeyPress>")
            self.root.unbind_all("<KeyRelease>")
            if self.capture_audio_stop_event is not None:
                self.capture_audio_stop_event.set()
            if self.capture_audio_thread is not None:
                self.capture_audio_thread.join()

            self.capture_input_var.set("")
            self._start_text_capture(
                self.config.passphrase,
                mode="login",
                on_success=lambda features: self._handle_login_features(features),
                on_error=self.capture_error_callback
                if hasattr(self, "capture_error_callback") and self.capture_error_callback
                else (lambda exc: self._set_status(str(exc))),
            )
            return "break"

        self._finalize_text_capture()
        return "break"

    def authenticate(self) -> None:
        self.load_model()
        if self.model_loaded is None:
            return

        gmail_value = self.auth_gmail_var.get().strip()
        if not gmail_value:
            self._set_status("Gmail cannot be empty.")
            return

        self.email_var.set(gmail_value)
        self.config.smtp_recipient = gmail_value
        self.config.extra["test_mode"] = self.test_mode_var.get()

        self._hide_otp_interface()
        self._hide_login_success()
        self._open_auth_passphrase_dialog()

    def _handle_login_features(self, features) -> None:
        try:
            threshold_text = (
                self.auth_threshold_var.get().strip()
                if self.auth_threshold_var is not None
                else str(self.config.auth_threshold)
            )
            try:
                threshold = float(threshold_text)
            except ValueError:
                threshold = float(self.config.auth_threshold)
                if self.auth_threshold_var is not None:
                    self.auth_threshold_var.set(f"{threshold:.6f}")
            self.config.auth_threshold = threshold
            save_config(self.config, self.config_path)

            from keystroke_auth.modeling import score_with_artifacts

            score = score_with_artifacts(self.model_loaded, features)
            # print numeric score to terminal for debugging/inspection
            print(f"Model decision score: {score:.6f} (threshold: {threshold:.6f})")
            if hasattr(self, "auth_dialog_status_var") and self.auth_dialog_status_var is not None:
                self.auth_dialog_status_var.set(f"Score: {score:.6f} (threshold {threshold:.6f})")
            self._set_status(f"Model score: {score:.6f} (threshold {threshold:.6f})")

            prediction = 1 if score > threshold else -1
        except Exception as exc:
            self._set_status(f"Prediction failed: {exc}")
            if self.capture_input_entry is not None:
                self.capture_input_entry.configure(state="normal")
            return

        if prediction == 1:
            self.pending_otp = None
            self._hide_otp_interface()
            self._close_auth_dialogs()
            self._show_login_success("AI directly identified the user. Login successful.")
            self._set_status("Login successful.")
            return

        self._set_status("Model rejected input. Enabling 2FA...")
        self.pending_otp = generate_otp(self.config.otp_digits)

        email_recipient = self.email_var.get().strip()
        is_test_mode = self.test_mode_var.get() if hasattr(self, "test_mode_var") else False

        def send_task() -> None:
            self.config.smtp_recipient = email_recipient or self.config.smtp_recipient
            send_otp_email(self.config, self.pending_otp, test_mode=is_test_mode)

        def on_success(_) -> None:
            self._set_status(
                "Model rejected input. OTP interface opened. (Check terminal if in Test Mode)"
            )

        def on_error(exc) -> None:
            self._set_status(f"OTP fallback unavailable: {exc}")

        self._close_auth_dialogs()
        self._open_otp_dialog()
        self._run_background(send_task, on_success=on_success, on_error=on_error)

    def verify_otp(self) -> None:
        entered = self.auth_otp_entry.get().strip() if self.auth_otp_entry is not None else ""
        if not self.pending_otp:
            self._set_status("No OTP challenge is pending.")
            return

        if entered and entered == self.pending_otp:
            self.pending_otp = None
            self._hide_otp_interface()
            self._close_auth_dialogs()
            self._show_login_success("OTP verified. Login successful.")
            self._set_status("Login successful.")
            return

        if self.auth_otp_entry is not None:
            self.auth_otp_entry.delete(0, "end")
            self.auth_otp_entry.focus_set()
        self._set_status("OTP mismatch. Access denied.")


def launch_app() -> None:
    app = KeystrokeAuthApp()
    app.launch()
