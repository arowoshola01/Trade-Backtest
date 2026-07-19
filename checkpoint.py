"""
checkpoint.py
Save/resume support for backtest.py so a long run survives a power
outage, laptop sleep/shutdown, or manual interrupt without losing
already-completed work.

Two things get cached per config, both keyed by the config label:
  1. The raw candle pull (candles_<label>.csv) -- no point re-pulling
     8000+ candles on resume when Pass 1's input never changes.
  2. Progress through Pass 2 (checkpoint_<label>.json) -- which flagged
     bars have already been tick-replayed + scored, and the trades
     accumulated so far.

Writes are atomic (write to a .tmp file, then os.replace over the real
file) so a crash mid-write can't corrupt a checkpoint -- os.replace is
atomic on both POSIX and Windows.
"""

import json
import os

import pandas as pd

CHECKPOINT_DIR = "backtest_checkpoints"


def _ensure_dir():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def candles_cache_path(label: str) -> str:
    return os.path.join(CHECKPOINT_DIR, f"candles_{label}.csv")


def checkpoint_path(label: str) -> str:
    return os.path.join(CHECKPOINT_DIR, f"checkpoint_{label}.json")


def load_candles_cache(label: str):
    """Returns a DataFrame if a cache exists, else None."""
    path = candles_cache_path(label)
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def save_candles_cache(label: str, df: pd.DataFrame):
    _ensure_dir()
    path = candles_cache_path(label)
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def load_checkpoint(label: str):
    """
    Returns a dict: {flagged_bar_epochs, processed_bar_epochs, trades,
    skipped_bar_epochs, dropped_bar_epochs, cluster_trajectories, skipped_cluster_ids}
    or None if no checkpoint exists yet. Bar identity is the candle's
    OPEN EPOCH (a fixed point in time), not a positional DataFrame index --
    positions shift if the underlying candle pull is ever refreshed or
    extended, epochs never do. Trade 'outcomes' dicts are converted back
    from JSON's string keys to int duration-minute keys.
    """
    path = checkpoint_path(label)
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        state = json.load(f)
    for trade in state.get("trades", []):
        trade["outcomes"] = {int(k): v for k, v in trade["outcomes"].items()}
    # Ensure skipped_cluster_ids exists for backward compatibility
    if "skipped_cluster_ids" not in state:
        state["skipped_cluster_ids"] = []
    return state


def save_checkpoint(label: str, flagged_bar_epochs: list, processed_bar_epochs: list, trades: list,
                    skipped_bar_epochs: list | None = None, dropped_bar_epochs: list | None = None,
                    cluster_trajectories: dict | None = None, skipped_cluster_ids: list | None = None):
    """
    Atomic save. flagged_bar_epochs / processed_bar_epochs are lists of
    candle OPEN EPOCHS (not positional indices) -- see load_checkpoint
    docstring for why. Trade 'outcomes' dicts get their int keys
    stringified for JSON (JSON object keys must be strings).

    cluster_trajectories: {cluster_id (str): {"epochs": [...], "counts":
    [...], "qualifying": [...], "bar_label": {epoch(str): label}}} --
    raw pre/cluster/post tick-level signal trajectories, captured once
    per fresh cluster by capture_cluster_trajectories() in backtest.py.
    Referenced by individual trades via their "cluster_id" field.
    Purely for offline analysis (repaint_analysis.py); never used to
    change entry timing or win/loss scoring.

    skipped_cluster_ids: list of cluster IDs (strings) that were at the
    edge of available history and couldn't be captured. These will be
    retried on resume if the dataframe has grown.
    """
    _ensure_dir()
    serializable_trades = []
    for trade in trades:
        t = dict(trade)
        t["outcomes"] = {str(k): v for k, v in trade["outcomes"].items()}
        t["bar_epoch"] = int(trade["bar_epoch"])
        serializable_trades.append(t)

    serializable_trajectories = {}
    for cid, traj in (cluster_trajectories or {}).items():
        entry = {
            "epochs": [int(e) for e in traj["epochs"]],
            "counts": [int(c) for c in traj["counts"]],
            "qualifying": [bool(q) for q in traj["qualifying"]],
            "bar_label": {str(e): lbl for e, lbl in traj["bar_label"].items()},
        }
        # Carry forward cached ticks + window bounds so Pass 2 can reuse
        # them on resume instead of re-pulling from Deriv.
        if "ticks" in traj:
            entry["ticks"] = traj["ticks"]
        if "window_start" in traj:
            entry["window_start"] = traj["window_start"]
        if "window_end" in traj:
            entry["window_end"] = traj["window_end"]
        serializable_trajectories[str(cid)] = entry

    state = {
        "flagged_bar_epochs": [int(e) for e in flagged_bar_epochs],
        "processed_bar_epochs": [int(e) for e in processed_bar_epochs],
        "trades": serializable_trades,
        "skipped_bar_epochs": [int(e) for e in (skipped_bar_epochs or [])],
        "dropped_bar_epochs": [int(e) for e in (dropped_bar_epochs or [])],
        "cluster_trajectories": serializable_trajectories,
        "skipped_cluster_ids": [str(cid) for cid in (skipped_cluster_ids or [])],
    }

    path = checkpoint_path(label)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, path)


def clear_checkpoint(label: str):
    """Remove checkpoint + candle cache for a config -- use to force a fresh run."""
    for path in (checkpoint_path(label), candles_cache_path(label)):
        if os.path.exists(path):
            os.remove(path)
