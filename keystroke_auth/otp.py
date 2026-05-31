from __future__ import annotations

import secrets
import smtplib
from email.message import EmailMessage

from .config import AppConfig


def generate_otp(digits: int = 6) -> str:
    upper = 10**digits
    value = secrets.randbelow(upper)
    return f"{value:0{digits}d}"


def send_otp_email(
    config: AppConfig, otp_code: str, subject: str = "Login verification code", test_mode: bool = False
) -> None:
    if test_mode or config.extra.get("test_mode"):
        print(f"--- TEST MODE: OTP is {otp_code} (email skipped) ---")
        return

    if not config.smtp_username or not config.smtp_password or not config.smtp_recipient:
        raise ValueError(
            "SMTP credentials and recipient must be configured before sending OTP email"
        )

    message = EmailMessage()
    message["From"] = config.smtp_username
    message["To"] = config.smtp_recipient
    message["Subject"] = subject
    message.set_content(f"Your verification code is {otp_code}")

    with smtplib.SMTP(config.smtp_host, config.smtp_port) as client:
        client.starttls()
        client.login(config.smtp_username, config.smtp_password)
        client.send_message(message)
