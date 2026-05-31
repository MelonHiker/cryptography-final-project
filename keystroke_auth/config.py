from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path("config.json")


@dataclass(slots=True)
class AppConfig:
    sample_rate: int = 44100
    key_length: int = 4500
    calibrated: bool = False
    passphrase: str = "cryptography final project"
    histogram_bins: int = 40
    auth_threshold: float = 0.0
    collection_session_count: int = 4
    white_noise_std: float = 0.0015
    otp_digits: int = 6
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_recipient: str = ""
    model_path: str = "model.pkl"
    scaler_path: str = "scaler.pkl"
    dataset_path: str = "dataset.csv"
    extra: dict[str, Any] = field(default_factory=dict)


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    config_path = Path(path)
    if not config_path.exists():
        return AppConfig()

    data = json.loads(config_path.read_text(encoding="utf-8"))
    base = asdict(AppConfig())
    extra = dict(data) if isinstance(data, dict) else {}
    merged = {**base, **extra}
    known_fields = {field.name for field in AppConfig.__dataclass_fields__.values()}
    config_kwargs = {
        key: value for key, value in merged.items() if key in known_fields and key != "extra"
    }
    config = AppConfig(**config_kwargs)
    config.extra = {key: value for key, value in merged.items() if key not in known_fields}
    return config


def save_config(config: AppConfig, path: str | Path = DEFAULT_CONFIG_PATH) -> None:
    config_path = Path(path)
    payload = asdict(config)
    payload.update(config.extra)
    config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
