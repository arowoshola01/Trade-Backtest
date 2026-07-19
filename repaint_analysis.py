"""
repaint_analysis.py
Reads an existing candle cache + Pass 2 checkpoint (no network calls, no
re-running Pass 2), recomputes Pass 1 fresh on the cached candles, and
cross-references every trade's ACTUAL entry (what tick_replay found
intrabar) against what Pass 1 would show for that same bar AT CLOSE.
Also reads cluster_trajectories (see cluster_trajectory.py) to analyze
signal STABILITY around each trade: within-bar flicker, pre-window
build-up pattern, and post-window survival.

Three repaint types, since "repainted" isn't one thing:
  side_flip       -- bar-close decision is the OPPOSITE direction from
                     what was actually traded. The strongest signature of
                     an unstable/unreliable signal.
  category_shift  -- same direction, but the category set differs (e.g.
                     traded via 'fallback' intrabar, bar-close shows
                     'direct' instead -- the exact Bar 206 case from
                     early testing).
  signal_vanished -- bar shows NO signal at all by close, even though a
                     trade fired on it intrabar.
  none            -- exact match, no repaint detected.

Trajectory-based stability metrics (only available for trades whose
cluster_id resolves to a captured trajectory -- see cluster_trajectory.py):
  flicker_count      -- number of True->False transitions in the signal's
                         qualifying state WITHIN this trade's own bar,
                         after it first qualifies. 0 = stayed qualified
                         once triggered; 1+ = flickered on/off before close.
  pre_buildup_type    -- "spike" (count jumped sharply in the single tick
                         right before this bar), "gradual" (built up
                         steadily over several ticks), "flat" (little
                         change), or "no_data" (trade's cluster has no
                         earlier ticks available in the trajectory, e.g.
                         it's a middle/last bar with no cluster history
                         before it captured -- rare, see note below).
  post_survival_type  -- "immediate_collapse" (already unqualified on the
                         very first post-bar tick), "gradual_decay"
                         (stayed qualified for a while then faded),
                         "persists" (still qualified through most/all of
                         the post-window), or "no_data".

Note on pre/post slicing: rather than relying on the trajectory's
pre/cluster/post bar labels (which only mark position within the whole
CLUSTER, not which specific bar within a multi-bar cluster), each
trade's pre/post window is sliced directly by EPOCH relative to that
trade's own bar_epoch. This works correctly regardless of a bar's
position in its cluster: for a middle bar, "pre" naturally becomes the
preceding cluster bar's tail end (still meaningful: was the signal
already building?), and "post" becomes the next cluster bar's start.

Usage:
    python repaint_analysis.py [--dir backtest_checkpoints]

Safe to run at any time, including against a partial/in-progress
checkpoint -- it only reads files, never writes to the checkpoint or
touches the network. Partial data gives a partial (possibly
unrepresentative) picture, so treat an early look as a rough gut-check,
not a final read.
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import pandas as pd

import config
from pipeline import run_pipeline
from strategy import StrategyConfig
import checkpoint as ckpt


def classify_repaint(actual_side, actual_categories, close_side, close_categories):
    if close_side is None:
        return "signal_vanished"
    if close_side != actual_side:
        return "side_flip"
    if set(close_categories) != set(actual_categories):
        return "category_shift"
    return "none"


def compute_flicker_count(trajectory: dict, bar_epoch: int, granularity: int):
    """
    Counts True->False transitions in `qualifying` within [bar_epoch,
    bar_epoch + granularity), starting from the first True (the actual
    entry point) onward. Returns None if no trajectory data covers this
    bar's own window at all.
    """
    epochs = trajectory["epochs"]
    qualifying = trajectory["qualifying"]
    in_bar = [(e, q) for e, q in zip(epochs, qualifying) if bar_epoch <= e < bar_epoch + granularity]
    if not in_bar:
        return None

    # find first True (entry), count True->False transitions after that
    started = False
    flickers = 0
    prev = None
    for _, q in in_bar:
        if not started:
            if q:
                started = True
                prev = q
            continue
        if prev and not q:
            flickers += 1
        prev = q
    return flickers if started else None


def classify_pre_buildup(trajectory: dict, bar_epoch: int):
    """
    Looks at counts for epochs strictly before bar_epoch (all available
    in the trajectory, not just one bar's worth). Classifies the pattern
    leading into this trade's entry.
    """
    epochs = trajectory["epochs"]
    counts = trajectory["counts"]
    pre = [(e, c) for e, c in zip(epochs, counts) if e < bar_epoch]
    if len(pre) < 2:
        return "no_data"

    pre_counts = [c for _, c in pre]
    start_count = pre_counts[0]
    end_count = pre_counts[-1]
    last_jump = pre_counts[-1] - pre_counts[-2] if len(pre_counts) >= 2 else 0

    if end_count - start_count <= 0:
        return "flat"
    if last_jump >= 2:
        return "spike"
    # gradual: net increase spread over multiple ticks, no single big jump
    if end_count - start_count >= 2:
        return "gradual"
    return "flat"


def classify_post_survival(trajectory: dict, bar_epoch: int, granularity: int):
    """
    Looks at qualifying/counts for epochs at/after this bar's close.
    Classifies whether the signal persisted, decayed, or collapsed.
    """
    epochs = trajectory["epochs"]
    qualifying = trajectory["qualifying"]
    bar_close = bar_epoch + granularity
    post = [(e, q) for e, q in zip(epochs, qualifying) if e >= bar_close]
    if not post:
        return "no_data"

    post_qualifying = [q for _, q in post]
    if not post_qualifying[0]:
        return "immediate_collapse"

    # fraction of the post-window that stayed qualified
    persisted_frac = sum(post_qualifying) / len(post_qualifying)
    if persisted_frac >= 0.7:
        return "persists"
    return "gradual_decay"


def analyze_config(label: str, chart_timeframe: str, tf_cal_override, base_dir: Path):
    candles_path = base_dir / f"candles_{label}.csv"
    checkpoint_path = base_dir / f"checkpoint_{label}.json"

    if not candles_path.exists() or not checkpoint_path.exists():
        print(f"[{label}] Missing candle cache or checkpoint in {base_dir} -- skipping.")
        return None

    df = pd.read_csv(candles_path).sort_values("epoch").reset_index(drop=True)
    with open(checkpoint_path) as f:
        state = json.load(f)
    trades = state.get("trades", [])
    if not trades:
        print(f"[{label}] No trades in checkpoint yet -- skipping.")
        return None

    cluster_trajectories = state.get("cluster_trajectories", {})

    granularity, tf_cal = config.resolve_granularity_and_tf_cal(chart_timeframe, tf_cal_override)
    pipeline_kwargs = dict(
        tf_cal=tf_cal, movol_model=config.MOVOL_MODEL, seconds_per_bar=granularity,
        movol_lookback_hours=config.MOVOL_LOOKBACK_HOURS,
        strategy_cfg=StrategyConfig(threshold=config.SIGNAL_THRESHOLD),
    )
    out = run_pipeline(df.drop(columns=["epoch"]), **pipeline_kwargs)
    out["epoch"] = df["epoch"].values
    epoch_to_pos = {int(e): i for i, e in enumerate(df["epoch"].values)}

    rows = []
    for t in trades:
        bar_epoch = int(t["bar_epoch"])
        actual_side = t["side"]
        actual_categories = t["categories"]
        outcomes = {int(k): v for k, v in t["outcomes"].items()}

        pos = epoch_to_pos.get(bar_epoch)
        if pos is None:
            continue  # bar no longer in cache (shouldn't normally happen)

        row = out.iloc[pos]
        decision = row["decision"]
        if decision == "CALL":
            close_side, close_categories = "buy", row["buy_categories"]
        elif decision == "PUT":
            close_side, close_categories = "sell", row["sell_categories"]
        else:
            close_side, close_categories = None, []

        repaint_type = classify_repaint(actual_side, actual_categories, close_side, close_categories)

        # ── trajectory-based stability metrics, if available ──
        cluster_id = t.get("cluster_id")
        trajectory = cluster_trajectories.get(cluster_id) if cluster_id else None
        if trajectory is not None:
            flicker_count = compute_flicker_count(trajectory, bar_epoch, granularity)
            pre_buildup_type = classify_pre_buildup(trajectory, bar_epoch)
            post_survival_type = classify_post_survival(trajectory, bar_epoch, granularity)
        else:
            flicker_count, pre_buildup_type, post_survival_type = None, "no_data", "no_data"

        for dur_min, win in outcomes.items():
            rows.append({
                "config": label, "bar_epoch": bar_epoch, "duration_min": dur_min,
                "actual_side": actual_side, "actual_categories": ",".join(actual_categories),
                "close_side": close_side, "close_categories": ",".join(close_categories),
                "repaint_type": repaint_type, "win": win,
                "has_trajectory": trajectory is not None,
                "flicker_count": flicker_count,
                "flicker_bucket": ("no_data" if flicker_count is None
                                    else ("stable" if flicker_count == 0 else "flickered")),
                "pre_buildup_type": pre_buildup_type,
                "post_survival_type": post_survival_type,
            })

    return pd.DataFrame(rows)


def summarize_by(df: pd.DataFrame, column: str):
    summary_rows = []
    for (cfg, dur, val), group in df.groupby(["config", "duration_min", column]):
        wins = group["win"].sum()
        total = len(group)
        summary_rows.append({
            "config": cfg, "duration_min": dur, column: val,
            "trades": total, "wins": int(wins), "losses": total - int(wins),
            "win_rate_pct": round(100 * wins / total, 2) if total else None,
        })
    return pd.DataFrame(summary_rows).sort_values(["config", "duration_min", column])


def generate_report(base_dir: Path = None):
    """
    Runs the full repaint/trajectory analysis across every config in
    config.BACKTEST_CONFIGS and returns:
        {
          "full_df": DataFrame or None,
          "repaint_summary", "flicker_summary", "pre_summary", "post_summary": DataFrames,
          "report_text": str -- the same human-readable report the CLI prints,
          "has_data": bool,
        }
    Does not print or save anything itself -- callers (main() below, or
    backtest.py's auto-run) decide what to do with the result.
    """
    base_dir = base_dir or Path("backtest_checkpoints")

    all_dfs = []
    log_lines = []
    for label, chart_timeframe, tf_cal_override in config.BACKTEST_CONFIGS:
        log_lines.append(f"Analyzing {label}...")
        df = analyze_config(label, chart_timeframe, tf_cal_override, base_dir)
        if df is not None and len(df):
            all_dfs.append(df)

    if not all_dfs:
        return {"full_df": None, "has_data": False, "report_text": "\n".join(log_lines + ["No data to analyze."])}

    full_df = pd.concat(all_dfs, ignore_index=True)

    repaint_summary = summarize_by(full_df, "repaint_type")
    flicker_summary = summarize_by(full_df, "flicker_bucket")
    pre_summary = summarize_by(full_df, "pre_buildup_type")
    post_summary = summarize_by(full_df, "post_survival_type")

    pd.set_option("display.width", 200)

    unique_trades = full_df.drop_duplicates(["config", "bar_epoch"])
    traj_coverage = unique_trades.groupby("config")["has_trajectory"].agg(["sum", "count"])

    sections = [
        "\n".join(log_lines),
        "===== TRAJECTORY COVERAGE (unique trades with trajectory data available) =====",
        traj_coverage.to_string(),
        "\n===== REPAINT TYPE DISTRIBUTION (per config, unique trades) =====",
        unique_trades.groupby(["config", "repaint_type"]).size().to_string(),
        "\n===== WIN RATE BY REPAINT TYPE (per config, per duration) =====",
        repaint_summary.to_string(index=False),
        "\n===== WIN RATE BY WITHIN-BAR FLICKER (stable vs. flickered before entry-bar close) =====",
        flicker_summary.to_string(index=False),
        "\n===== WIN RATE BY PRE-WINDOW BUILD-UP PATTERN (spike vs. gradual vs. flat) =====",
        pre_summary.to_string(index=False),
        "\n===== WIN RATE BY POST-WINDOW SURVIVAL (persists vs. decays vs. collapses) =====",
        post_summary.to_string(index=False),
    ]
    report_text = "\n".join(sections)

    return {
        "full_df": full_df, "has_data": True, "report_text": report_text,
        "repaint_summary": repaint_summary, "flicker_summary": flicker_summary,
        "pre_summary": pre_summary, "post_summary": post_summary,
    }


def main():
    parser = argparse.ArgumentParser(description="Detect and analyze signal repaints in backtest checkpoints")
    parser.add_argument("--dir", default="backtest_checkpoints", help="Directory containing checkpoints")
    args = parser.parse_args()
    base_dir = Path(args.dir)

    result = generate_report(base_dir)
    print(result["report_text"])

    if not result["has_data"]:
        return

    result["full_df"].to_csv("repaint_analysis_detail.csv", index=False)
    result["repaint_summary"].to_csv("repaint_analysis_summary.csv", index=False)
    result["flicker_summary"].to_csv("repaint_analysis_flicker_summary.csv", index=False)
    result["pre_summary"].to_csv("repaint_analysis_pre_buildup_summary.csv", index=False)
    result["post_summary"].to_csv("repaint_analysis_post_survival_summary.csv", index=False)

    print("\nSaved: repaint_analysis_detail.csv, repaint_analysis_summary.csv, "
          "repaint_analysis_flicker_summary.csv, repaint_analysis_pre_buildup_summary.csv, "
          "repaint_analysis_post_survival_summary.csv")


if __name__ == "__main__":
    main()
