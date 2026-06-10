# Keystroke & Acoustic Multi-Modal Authentication System

This project implements a multi-modal biometric identity authentication system using **Keystroke Dynamics** combined with **Acoustic Signals (Key-Sound Analysis)**, inspired by the academic paper: _"User Identification and Authentication Using Keystroke Dynamics with Acoustic Signal"_.

The system captures physical timing (dwell time, flight time) alongside microphones capture of key-press sounds, extracting a **46-dimensional feature vector** to feed a **One-Class Support Vector Machine (OC-SVM)** for secure, non-intrusive continuous authentication.

---

## 🏗️ System Architecture

The project consists of the following key modules:

- `main.py`: The main entry point of the application.
- `keystroke_auth/`: Core library containing all components:
  - `gui.py`: The dashboard implemented using **CustomTkinter**, featuring a modern dark-theme user interface.
  - `capture.py`: Low-level background hooks for multi-threaded audio (`pyaudio`) and global key-event tracking (`pynput`).
  - `features.py`: Advanced feature-engineering pipeline extracting timing statistics and 13-frame acoustic Mel-Frequency Cepstral Coefficients (MFCCs).
  - `model.py`: OC-SVM pipeline (with RBF Kernel) handles dataset formatting, hyperparameter scaling, and training.
  - `otp.py`: SMTP-based multi-factor authentication (MFA) fallback system.
  - `config.py`: Local JSON state-management for credentials and hardware profiles.

---

## ⚙️ Environment Setup & Permissions

### 1. System Dependencies (macOS)

`pyaudio` depends on system-level `portaudio` libraries. Please install them first:

```bash
brew install portaudio
pip install -r requirements.txt
```

### 2. Required Operating System Permissions (CRITICAL)

Because this app runs low-level global input listeners and audio recording hooks, you **MUST** grant permissions to your terminal or IDE:

- **Microphone Access**: Required by PyAudio to capture key-press sounds during typing.
- **Accessibility Permissions**: Required by `pynput` to log keyboard release/press timings globally.
  - _How to grant_: Open `System Settings` -> `Privacy & Security` -> `Accessibility` -> Toggle **ON** your terminal (e.g., Terminal, iTerm2) or IDE (e.g., VS Code). If it is already ON, toggle it OFF and ON again to refresh permissions.

---

## 🚀 Step-by-Step Guide

### 📂 Step 0: Configure SMTP for 2FA Fallback

To enable the One-Time Password (OTP) fallback when the AI model rejects a valid passphrase, configure your email credentials:

1.  Open the **Settings** tab.
2.  Input the following details:
    - **SMTP Host**: `smtp.gmail.com` (Default for Gmail).
    - **SMTP Port**: `587` (Default for TLS).
    - **SMTP Sender Username**: Your Gmail address (e.g., `yourname@gmail.com`).
    - **SMTP Sender Password**: A **Google App Password** (not your regular login password).
      > [!TIP]
      > **How to get a Google App Password**:
      > Go to your Google Account -> _Security_ -> _2-Step Verification_ -> Scroll to the bottom to _App Passwords_ -> Generate a password named "Keystroke App" and copy the 16-character code.
3.  Click **Save Settings** to write them into `config.json`.

---

### 🏋️ Training Pipeline

1.  **Stage 1: Calibration**
    - Start the app: `python main.py`.
    - Click **Start Calibration** and tap the **Space Bar 3 times**.
    - The app measures your acoustic key-sound decay speed to determine a personalized audio frame length and saves it dynamically.
2.  **Stage 2: Passphrase**
    - Input your secret passphrase (or keep the default one).
    - Click **Save Passphrase to config.json**. The GUI will automatically switch to **Stage 3**.
3.  **Stage 3: Data Collection**
    - Enter the desired sample size.
    - Enter the **Number of sessions** to divide the data collection into (this introduces periodic pauses, which encourages natural variations in typing rhythm and environmental noise).
    - Click **Start Data Collection** to trigger the floating enrollment window.
    - Type the passphrase exactly as prompted, pressing `Enter` to submit. Repeat until complete. The results are saved into `dataset.csv`.
4.  **Stage 4: Modeling**
    - Verify the path to your `dataset.csv`.
    - Select the **Algorithm** (`lof`, `iforest`, `pca_svm`, or `oc_svm`).
    - Tune the hyperparameters:
      - **Contamination / `nu`** ($\nu$): Controls the training error upper bound or the proportion of outliers in the training set (defines how strict the boundary is).
      - **Gamma** ($\gamma$): Controls the RBF kernel influence scale (used by SVM-based algorithms).
      - **N Estimators**: Number of trees in the forest (used by `iforest`).
      - **N Components**: Number of principal components to retain (used by `pca_svm`).
    - Click **Train Model From CSV**. The trained weights are exported as `model.pkl` and `scaler.pkl`.

---

### 🧠 Model Tuning & Training Precautions

Due to the nature of high-dimensional feature spaces (46-D) and typical small sample sizes, the One-Class SVM can easily **overfit**, resulting in a decision boundary that is too strict. If you find the model rejecting your input almost every time in Stage 5, follow these critical tuning steps:

> [!TIP]
> **Loosen the Decision Boundary (Stage 4 Tuning)**
>
> - **Lower `nu` ($\nu$)**: The default `nu=0.1` forces the model to treat 10% of your training data as outliers. Change `nu` to **`0.01`** or **`0.005`** to make the model more inclusive of all your training data.
> - **Lower `gamma` ($\gamma$)**: The default `scale` can sometimes compute a value that makes the RBF kernel's influence radius too narrow (creating "isolated islands" of acceptance). Manually set `gamma` to a smaller float like **`0.001`** or **`0.0005`** to create a wider, smoother decision landscape.

> [!IMPORTANT]
> **Data Collection Consistency (Stage 3)**
>
> - **Typing Consistency**: It doesn't matter if you type fast or slow, what matters is **consistency**. Try to type at your most natural, repeatable rhythm. If your typing speed fluctuates wildly during data collection, the feature variance will explode.
> - **Environmental Noise**: 32 of the 46 features are acoustic (MFCCs). Record your samples in a **quiet environment**. Air conditioners blowing directly into the mic, breathing heavily, or people talking in the background will drastically alter the acoustic features and cause false rejections.

---

### 🔐 Testing & Authentication

1.  Navigate to **Stage 5: Authentication**.
2.  Ensure that **Model Path** and **Scaler Path** point to your exported `.pkl` files.
3.  In **Recipient Gmail**, enter the receiver's address where the OTP code should be delivered.
4.  _(Optional)_ Toggle **Test Mode** if you want to test the entire validation pipeline (including OTP entry) without sending real emails (this prevents being throttled or banned by Gmail SMTP limits during rapid testing).
5.  _(Optional)_ Adjust the **Auth Threshold**. A higher threshold makes the system stricter, increasing security but potentially raising the False Rejection Rate (FRR).
6.  Click **Authenticate**. A specialized, secure login window will pop up.
7.  Begin typing your passphrase. The system dynamically records your acoustic waveforms and keystroke dynamics in real-time. Press `Enter` to log in.
8.  **Outcomes**:
    - **Success**: If both typing speed/sounds and the characters match, a large, prominent green **Login Success** window pops up.
    - **Incorrect Passphrase**: If you mistype, the passphrase window displays a red error warning inline and resets the text box, letting you retry instantly.
    - **Passphrase Correct but AI Model Rejects (MFA Fallback)**: If the text is typed correctly but the SVM model determines the typing dynamics or sounds do not match the registered owner, a dedicated **OTP Verification popup** appears, triggering a 6-digit OTP code to the configured Gmail recipient. Enter the code to bypass security.

---

## 📝 Configuration File (`config.json`)

The application state is entirely managed locally in `config.json`. Do not check this file into Git repositories, as it contains sensitive App Passwords.

---

## ?? Attack Simulation & Validation

This project also includes an independent attack-simulation toolkit for validating how well the trained authentication model resists weak black-box Hydra-style spoofing attempts while still preserving the original **keystroke-acoustic analysis** workflow. These simulations do not assume access to the model internals, do not use owner training distributions to generate attacks, and do not use acoustic replay recordings.

### ?? Launch the Independent Attack-Simulation GUI

After you have already collected training data and exported `model.pkl` + `scaler.pkl`, you can launch the standalone validation interface:

```bash
python run_attack_simulation.py
```

The GUI reads the following artifacts by default:

- `dataset.csv`
- `model.pkl`
- `scaler.pkl`
- `config.json`

It then runs the built-in black-box Hydra-style attack scenarios and summarizes:

- **Accepted attempts**
- **Acceptance rate**
- **Joint model score range**
- **Acoustic / timing feature-distance statistics**

---

### ?? Run the Attack Script from the Command Line

If you prefer a scriptable workflow, use the CLI entry point:

```bash
python -m attack_simulator.cli --attempts 250 --output attack_simulation_report.json
```

This runs all supported attacks and saves the report locally.

To run only a specific attack:

```bash
python -m attack_simulator.cli --attack hydra_humanized_timing --attempts 250 --output hydra_humanized_report.csv
```

Available attack names include:

- `hydra_scripted_burst`
- `hydra_humanized_timing`
- `hydra_synthetic_keyboard`

You can also select the realism profile:

```bash
python -m attack_simulator.cli --realism casual --attempts 250
```

---

### ?? Recommended Validation Workflow

1.  Complete **Stage 1 - Stage 4** in the main GUI and export `dataset.csv`, `model.pkl`, and `scaler.pkl`.
2.  Run either the standalone GUI (`python run_attack_simulation.py`) or the CLI (`python -m attack_simulator.cli ...`).
3.  Review the **Acceptance Rate** of each attack scenario.
4.  If you have updated the model-defense logic, **retrain the model first** before comparing new attack-simulation results.

> [!IMPORTANT]
> The attack-simulation toolkit evaluates the currently trained artifacts. If `model.pkl` and `scaler.pkl` were produced before a defense update, retrain them in **Stage 4 Modeling** before drawing conclusions from the new report.
