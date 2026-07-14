import html
import mimetypes
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


def format_table(rows: list[list[str]], headers: list[str]) -> str:
    if rows:
        col_widths = [max(len(str(item)) for item in column) for column in zip(headers, *rows)]
    else:
        col_widths = [len(h) for h in headers]
    header_row = " | ".join(header.ljust(width) for header, width in zip(headers, col_widths))
    separator = "-+-".join("-" * width for width in col_widths)
    body_rows = [" | ".join(str(value).ljust(width) for value, width in zip(row, col_widths)) for row in rows]
    return "\n".join([header_row, separator] + body_rows)


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
    elif event == "analysis":
        subject = f"[Backtest] {label} analysis summary"
        body = (
            f"Checkpoint analysis completed for {label}.\n"
            f"Processed bars: {progress_count}/{total}\n"
            f"Trades saved: {trades}\n"
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

    return send_raw_email(subject, body)


def build_html_body(body: str) -> str:
    escaped = html.escape(body)
    return f"<html><body><pre style='font-family:Menlo,Monaco,Consolas,monospace; white-space:pre-wrap;'>" + escaped + "</pre></body></html>"


def send_raw_email(subject: str, body: str) -> bool:
    if not is_configured():
        return False

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
    msg.add_alternative(build_html_body(body), subtype="html")

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


def send_email_with_attachments(subject: str, body: str, attachment_paths: list[str]) -> bool:
    if not is_configured():
        return False

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
    msg.add_alternative(build_html_body(body), subtype="html")

    for path in attachment_paths:
        try:
            with open(path, "rb") as f:
                data = f.read()
            ctype, encoding = mimetypes.guess_type(path)
            if ctype:
                maintype, subtype = ctype.split("/", 1)
            else:
                maintype, subtype = "application", "octet-stream"
            msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=os.path.basename(path))
        except Exception as exc:
            print(f"[email] failed to attach {path}: {exc}")
            return False

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


def build_summary_message(items: list[tuple], event: str) -> tuple[str, str]:
    if event == "done":
        subject = "[Backtest] aggregate run completed"
        header = "Backtest run completed."
    else:
        subject = "[Backtest] aggregate checkpoint progress update"
        header = "Backtest checkpoint progress update."

    rows = []
    for label, processed, total, trades in items:
        percent = (processed / total * 100.0) if total else 100.0
        rows.append([label, f"{processed}/{total}", f"{percent:.1f}%", str(trades)])

    body = "\n".join([
        header,
        "",
        format_table(rows, ["config", "processed/total", "progress", "trades"]),
    ])
    return subject, body


def send_summary_email(items: list[tuple], event: str = "checkpoint") -> bool:
    if not is_configured():
        return False

    subject, body = build_summary_message(items, event)
    return send_raw_email(subject, body)


def build_final_results_message(combo_body: str, category_body: str) -> tuple[str, str]:
    subject = "[Backtest] final run results"
    body_lines = [
        "===== COMBO RESULTS (mutually exclusive, sums to total trades) =====",
        combo_body,
        "",
        "===== PER-CATEGORY RESULTS (inclusive, overlaps counted in each) =====",
        category_body,
    ]
    return subject, "\n".join(body_lines)


def send_final_results_email(combo_body: str, category_body: str) -> bool:
    if not is_configured():
        return False

    subject, body = build_final_results_message(combo_body, category_body)
    return send_raw_email(subject, body)


def build_analysis_summary_message(items: list[dict]) -> tuple[str, str]:
    subject = "[Backtest] analyze checkpoints summary"
    rows = []
    detail_lines = []
    for item in items:
        rows.append([
            item['label'],
            f"{item['processed']}/{item['total']}",
            f"{item['progress']:.1f}%",
            str(item['trades']),
        ])

        item_details = []
        if item.get('combo_counts'):
            item_details.append(f"{item['label']} combos:")
            for combo, count in item['combo_counts'].items():
                item_details.append(f"  {combo}: {count}")
        if item.get('category_counts'):
            item_details.append(f"{item['label']} categories:")
            for category, count in item['category_counts'].items():
                item_details.append(f"  {category}: {count}")

        if item_details:
            detail_lines.extend(item_details + [""])

    body_lines = [
        "Checkpoint analysis summary.",
        "",
        format_table(rows, ["file", "processed/total", "progress", "trades"]),
    ]
    if detail_lines:
        body_lines.extend(["", "Details:", ""] + detail_lines)

    return subject, "\n".join(body_lines)


def send_analysis_summary_email(items: list[dict]) -> bool:
    if not is_configured():
        return False

    subject, body = build_analysis_summary_message(items)
    return send_raw_email(subject, body)
