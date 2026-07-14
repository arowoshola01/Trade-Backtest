#!/usr/bin/env python3
"""Show current backtest progress from checkpoint files.

Examples:
    python check_backtest_status.py
    python check_backtest_status.py --label 5m_auto_M5
    python check_backtest_status.py --watch 30
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import checkpoint as ckpt


def discover_labels(checkpoint_dir: str | None = None) -> list[str]:
    base_dir = Path(checkpoint_dir or ckpt.CHECKPOINT_DIR)
    if not base_dir.exists():
        return []

    labels = []
    for path in sorted(base_dir.glob("checkpoint_*.json")):
        label = path.name[len("checkpoint_") : -len(".json")]
        labels.append(label)
    return labels


def is_backtest_running() -> bool:
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid=,args="],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return False

    text = result.stdout or ""
    return "backtest.py" in text


def format_status(label: str) -> str:
    state = ckpt.load_checkpoint(label)
    candle_path = ckpt.candles_cache_path(label)
    checkpoint_path = ckpt.checkpoint_path(label)

    if state is None:
        return (
            f"[{label}] No checkpoint found yet.\n"
            f"  checkpoint file: {'present' if os.path.exists(checkpoint_path) else 'missing'}\n"
            f"  candle cache   : {'present' if os.path.exists(candle_path) else 'missing'}\n"
            f"  running        : {'yes' if is_backtest_running() else 'no'}"
        )

    flagged = state.get("flagged_bar_epochs", [])
    processed = state.get("processed_bar_epochs", [])
    skipped = state.get("skipped_bar_epochs", [])
    dropped = state.get("dropped_bar_epochs", [])
    trades = state.get("trades", [])

    total = len(flagged)
    done = len(processed)
    remaining = max(total - done, 0)
    percent = (done / total * 100.0) if total else 100.0

    lines = [
        f"[{label}]",
        f"  checkpoint file: present",
        f"  candle cache   : {'present' if os.path.exists(candle_path) else 'missing'}",
        f"  running        : {'yes' if is_backtest_running() else 'no'}",
        f"  flagged bars   : {total}",
        f"  processed      : {done}",
        f"  remaining      : {remaining}",
        f"  progress       : {percent:.1f}%",
        f"  skipped bars   : {len(skipped)}",
        f"  dropped bars   : {len(dropped)}",
        f"  trades         : {len(trades)}",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Show backtest progress from checkpoint files")
    parser.add_argument("--label", help="Show status for one config label")
    parser.add_argument("--watch", type=int, default=0, help="Refresh every N seconds")
    args = parser.parse_args()

    if args.watch and args.watch <= 0:
        parser.error("--watch requires a positive integer")

    while True:
        if args.label:
            labels = [args.label]
        else:
            labels = discover_labels()

        if not labels:
            print("No backtest checkpoint files found in backtest_checkpoints/.")
            return 1

        for label in labels:
            print(format_status(label))
            print()

        if not args.watch:
            return 0

        time.sleep(args.watch)


if __name__ == "__main__":
    sys.exit(main())
