# Keystroke & Acoustic Multi-Modal Authentication System

This project implements a multi-modal biometric identity authentication system using **Keystroke Dynamics** combined with **Acoustic Signals (Key-Sound Analysis)**, inspired by the academic paper: *"User Identification and Authentication Using Keystroke Dynamics with Acoustic Signal"*.

The system captures physical timing (dwell time, flight time) alongside microphones capture of key-press sounds, extracting a **46-dimensional feature vector** to feed a **One-Class Support Vector Machine (OC-SVM)** for secure, non-intrusive continuous authentication.

---

## 🏗️ System Architecture

The project consists of the following key modules:
*   `main.py`: The main entry point of the application.
*   `keystroke_auth/`: Core library containing all components:
    *   `gui.py`: The dashboard implemented using **CustomTkinter**, featuring a modern dark-theme user interface.
    *   `capture.py`: Low-level background hooks for multi-threaded audio (`pyaudio`) and global key-event tracking (`pynput`).
    *   `features.py`: Advanced feature-engineering pipeline extracting timing statistics and 13-frame acoustic Mel-Frequency Cepstral Coefficients (MFCCs).
    *   `model.py`: OC-SVM pipeline (with RBF Kernel) handles dataset formatting, hyperparameter scaling, and training.
    *   `otp.py`: SMTP-based multi-factor authentication (MFA) fallback system.
    *   `config.py`: Local JSON state-management for credentials and hardware profiles.

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
*   **Microphone Access**: Required by PyAudio to capture key-press sounds during typing.
*   **Accessibility Permissions**: Required by `pynput` to log keyboard release/press timings globally.
    *   *How to grant*: Open `System Settings` -> `Privacy & Security` -> `Accessibility` -> Toggle **ON** your terminal (e.g., Terminal, iTerm2) or IDE (e.g., VS Code). If it is already ON, toggle it OFF and ON again to refresh permissions.

---

## 🚀 Step-by-Step Guide

### 📂 Step 0: Configure SMTP for 2FA Fallback
To enable the One-Time Password (OTP) fallback when the AI model rejects a valid passphrase, configure your email credentials:
1.  Open the **Settings** tab.
2.  Input the following details:
    *   **SMTP Host**: `smtp.gmail.com` (Default for Gmail).
    *   **SMTP Port**: `587` (Default for TLS).
    *   **SMTP Sender Username**: Your Gmail address (e.g., `yourname@gmail.com`).
    *   **SMTP Sender Password**: A **Google App Password** (not your regular login password).
        > [!TIP]
        > **How to get a Google App Password**:
        > Go to your Google Account -> *Security* -> *2-Step Verification* -> Scroll to the bottom to *App Passwords* -> Generate a password named "Keystroke App" and copy the 16-character code.
3.  Click **Save Settings** to write them into `config.json`.

---

### 🏋️ Training Pipeline

1.  **Stage 1: Calibration**
    *   Start the app: `python main.py`.
    *   Click **Start Calibration** and tap the **Space Bar 3 times**.
    *   The app measures your acoustic key-sound decay speed to determine a personalized audio frame length and saves it dynamically.
2.  **Stage 2: Passphrase**
    *   Input your secret passphrase (or keep the default one).
    *   Click **Save Passphrase to config.json**. The GUI will automatically switch to **Stage 3**.
3.  **Stage 3: Data Collection**
    *   Enter the desired sample size (we recommend **10–15 samples** for robust SVM boundary training).
    *   Click **Start Data Collection** to trigger the floating enrollment window.
    *   Type the passphrase exactly as prompted, pressing `Enter` to submit. Repeat until complete. The results are saved into `dataset.csv`.
4.  **Stage 4: Modeling**
    *   Verify the path to your `dataset.csv`.
    *   Tune the hyperparameters:
        *   `nu` ($\nu$): Controls the training error upper bound (defines how strict the boundary is).
        *   `gamma` ($\gamma$): Controls the RBF kernel influence scale.
    *   Click **Train Model From CSV**. The trained weights are exported as `model.pkl` and `scaler.pkl`.

---

### 🔐 Testing & Authentication

1.  Navigate to **Stage 5: Authentication**.
2.  Ensure that **Model Path** and **Scaler Path** point to your exported `.pkl` files.
3.  In **Recipient Gmail**, enter the receiver's address where the OTP code should be delivered.
4.  *(Optional)* Toggle **Test Mode** if you want to test the entire validation pipeline (including OTP entry) without sending real emails (this prevents being throttled or banned by Gmail SMTP limits during rapid testing).
5.  Click **Authenticate**. A specialized, secure login window will pop up.
6.  Begin typing your passphrase. The system dynamically records your acoustic waveforms and keystroke dynamics in real-time. Press `Enter` to log in.
7.  **Outcomes**:
    *   **Success**: If both typing speed/sounds and the characters match, a large, prominent green **Login Success** window pops up.
    *   **Incorrect Passphrase**: If you mistype, the passphrase window displays a red error warning inline and resets the text box, letting you retry instantly.
    *   **Passphrase Correct but AI Model Rejects (MFA Fallback)**: If the text is typed correctly but the SVM model determines the typing dynamics or sounds do not match the registered owner, a dedicated **OTP Verification popup** appears, triggering a 6-digit OTP code to the configured Gmail recipient. Enter the code to bypass security.

---

## 📝 Configuration File (`config.json`)

The application state is entirely managed locally in `config.json`. Do not check this file into Git repositories, as it contains sensitive App Passwords.

```json
{
  "key_sound_len": 4410,
  "passphrase": "your_secret_passphrase",
  "smtp_host": "smtp.gmail.com",
  "smtp_port": 587,
  "smtp_username": "sender@gmail.com",
  "smtp_password": "xxxx xxxx xxxx xxxx",
  "smtp_recipient": "recipient@gmail.com"
}
```

