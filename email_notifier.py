import os
import smtplib
from email.message import EmailMessage


def is_configured() -> bool:
    return bool(
        os.getenv("BACKTEST_SMTP_HOST")
        and os.getenv("BACKTEST_SMTP_FROM")
        and os.getenv("BACKTEST_SMTP_TO")
    )


def should_notify(progress_count: int, notify_every: int = 50) -> bool:
    return progress_count > 0 and progress_count % notify_every == 0


def build_message(label: str, progress_count: int, total: int, trades: int, event: str) -> str:
    percent = (progress_count / total * 100.0) if total else 100.0
    if event == "done":
        subject = f"[Backtest] {label} completed"
        body = (
            f"Backtest completed for {label}.\n"
            f"Processed bars: {progress_count}/{total}\n"
            f"Trades recorded: {trades}\n"
            f"Progress: {percent:.1f}%"
        )
    else:
        subject = f"[Backtest] {label} progress update"
        body = (
            f"Backtest update for {label}.\n"
            f"Processed bars: {progress_count}/{total}\n"
            f"Trades recorded: {trades}\n"
            f"Progress: {percent:.1f}%"
        )
    return subject, body


def send_email(label: str, progress_count: int, total: int, trades: int, event: str = "checkpoint") -> bool:
    if not is_configured():
        return False

    subject, body = build_message(label, progress_count, total, trades, event)

    smtp_host = os.getenv("BACKTEST_SMTP_HOST")
    smtp_port = int(os.getenv("BACKTEST_SMTP_PORT", "587"))
    smtp_user = os.getenv("BACKTEST_SMTP_USERNAME")
    smtp_password = os.getenv("BACKTEST_SMTP_PASSWORD")
    smtp_from = os.getenv("BACKTEST_SMTP_FROM")
    smtp_to = os.getenv("BACKTEST_SMTP_TO")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = smtp_from
    msg["To"] = smtp_to
    msg.set_content(body)

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            if smtp_user and smtp_password:
                server.login(smtp_user, smtp_password)
            server.send_message(msg)
        return True
    except Exception as exc:
        print(f"[email] failed to send notification: {exc}")
        return False
