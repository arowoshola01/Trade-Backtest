"""
tick_replay.py
Pass 2 of the two-pass backtest: for a bar that Pass 1 flagged as CALL/PUT
at bar-close resolution, replay that bar's ticks (starting from its OPEN)
one at a time, building the forming bar's OHLC as each tick arrives, and
find the exact tick where the signal condition first turned true for the
flagged side -- matching what a live tick-by-tick bot would have seen.

IMPORTANT APPROXIMATION: re-running the full pipeline (all indicators +
both HMM forward filters + CISD's backward-scanning trailer) from a cold
start on every single tick, using the ENTIRE preceding history, would be
far too slow for a backtest with hundreds of flagged bars x hundreds of
ticks each. Instead, each tick's pipeline run uses a bounded trailing
WARMUP_BARS window of already-closed bars (default 300) ahead of the
forming bar, not the full dataset. This is enough for every rolling
indicator (longest lookback in the suite is the 35-bar z-score window)
and gives CISD's trailers a reasonable amount of run-up room, but it is
NOT a bit-for-bit replica of running the indicator from the very start of
history -- CISD in particular could anchor to a slightly different origin
level than a full-history run would produce, in the (rare) case the true
origin sits further back than the warmup window. Flagging this clearly
rather than silently presenting it as exact.
"""

import pandas as pd
import numpy as np
import asyncio

from pipeline import run_pipeline

WARMUP_BARS = 300


def build_warmup_history(full_df: pd.DataFrame, bar_index: int, warmup_bars: int = WARMUP_BARS) -> pd.DataFrame:
    """
    Returns the trailing window of already-CLOSED bars immediately
    preceding `bar_index`, to use as fixed context for tick replay.
    Does not include bar_index itself (that's the bar being replayed).
    """
    start = max(0, bar_index - warmup_bars)
    return full_df.iloc[start:bar_index][["open", "high", "low", "close", "volume"]].reset_index(drop=True)


def iter_bar_ticks(ticks: list, bar_open_epoch: int, bar_close_epoch: int):
    """
    ticks: list of {epoch, price} sorted oldest->newest, covering at least
    [bar_open_epoch, bar_close_epoch).
    Yields (epoch, price) for ticks that fall within this bar's window.
    """
    for t in ticks:
        if bar_open_epoch <= t["epoch"] < bar_close_epoch:
            yield t["epoch"], t["price"]


def find_entry_tick(full_df: pd.DataFrame, bar_index: int, ticks_in_bar: list,
                     side: str, pipeline_kwargs: dict, warmup_bars: int = WARMUP_BARS):
    """
    ticks_in_bar: list of (epoch, price) tuples, oldest first, already
    filtered to this bar's window (e.g. via list(iter_bar_ticks(...))).

    Replays them against the warmup history, appending a partial
    forming-bar row that updates on each tick, and re-running the
    pipeline after each tick. Returns the first tick where `side` ('buy'
    or 'sell') qualifies under ANY category, as a dict:
        {epoch, price, categories, open, high, low, close}
    or None if the side never qualifies across all ticks in the bar
    (shouldn't normally happen if Pass 1 flagged this bar/side at close,
    but data gaps or partial tick coverage can cause it -- caller should
    treat None as "drop this signal, insufficient tick coverage").
    """
    if not ticks_in_bar:
        return None

    warmup = build_warmup_history(full_df, bar_index, warmup_bars)
    if len(warmup) < 40:
        # not enough context for the longer rolling windows (z-score
        # period 35, etc.) to produce meaningful values -- skip
        return None

    bar_open_price = ticks_in_bar[0][1]
    running_high = bar_open_price
    running_low = bar_open_price

    import os
    debug = os.environ.get("TICK_REPLAY_DEBUG")

    for idx, (epoch, price) in enumerate(ticks_in_bar):
        running_high = max(running_high, price)
        running_low = min(running_low, price)
        partial_row = pd.DataFrame([{
            "open": bar_open_price, "high": running_high,
            "low": running_low, "close": price, "volume": 1.0,
        }])
        working_df = pd.concat([warmup, partial_row], ignore_index=True)

        try:
            out = run_pipeline(working_df, **pipeline_kwargs)
        except Exception as e:
            if debug:
                print(f"idx={idx} EXCEPTION: {e!r}")
            continue

        last = out.iloc[-1]
        categories = last["buy_categories"] if side == "buy" else last["sell_categories"]
        if debug:
            print(f"idx={idx} categories={categories} warmup_len={len(warmup)} working_len={len(working_df)}")
        if categories:
            return {
                "epoch": epoch, "price": price, "categories": categories,
                "open": bar_open_price, "high": running_high, "low": running_low, "close": price,
            }

    return None


def find_settlement_price(ticks_after_entry: list, entry_epoch: int, duration_seconds: int):
    """
    ticks_after_entry: list of {epoch, price}, sorted oldest->newest,
    covering at least [entry_epoch, entry_epoch + duration_seconds].
    Returns the price of the first tick at or after
    (entry_epoch + duration_seconds), or None if no such tick is present
    (caller should drop the trade -- insufficient tick coverage / edge of
    pulled data).
    """
    target_epoch = entry_epoch + duration_seconds
    for t in ticks_after_entry:
        if t["epoch"] >= target_epoch:
            return t["price"], t["epoch"]
    return None, None


# ═══════════════════════════════════════════════════════════════════════════
# CLUSTER TRAJECTORY TRACKING
#   Groups consecutive same-direction flagged bars into one cluster, pulls
#   exactly one pre-window bar before it and one post-window bar after it,
#   and tracks the same-side signal count CONTINUOUSLY across the whole
#   span -- not just the first-qualifying-tick snapshot find_entry_tick
#   uses for scoring. This is purely for repaint/stability ANALYSIS
#   (consumed by repaint_analysis.py); it does not change entry timing or
#   win/loss scoring, which still comes from find_entry_tick +
#   find_settlement_price as before.
# ═══════════════════════════════════════════════════════════════════════════

def cluster_flagged_bars(epoch_side_pairs: list, granularity: int):
    """
    epoch_side_pairs: list of (epoch, side) tuples for flagged bars,
    side is 'buy' or 'sell'.

    Groups epochs into clusters of CONSECUTIVE bars (epoch2 == epoch1 +
    granularity) that also share the SAME side -- a direction change
    always breaks a cluster, since there's no single side left to track
    continuously across it. An isolated flagged bar with no matching
    neighbor becomes its own single-bar cluster.

    Returns a list of dicts: {"epochs": [...], "side": "buy"/"sell"},
    sorted oldest -> newest, each internally sorted oldest -> newest.
    """
    pairs = sorted(epoch_side_pairs, key=lambda p: p[0])
    clusters = []
    current = None
    for epoch, side in pairs:
        if current is not None and epoch == current["epochs"][-1] + granularity and side == current["side"]:
            current["epochs"].append(epoch)
        else:
            if current is not None:
                clusters.append(current)
            current = {"epochs": [epoch], "side": side}
    if current is not None:
        clusters.append(current)
    return clusters


async def compute_cluster_trajectory(full_df: pd.DataFrame, cluster_first_bar_index: int, num_cluster_bars: int,
                                ticks: list, side: str, granularity: int, pipeline_kwargs: dict,
                                warmup_bars: int = WARMUP_BARS):
    """
    Walks `ticks` (list of (epoch, price) tuples, oldest first, covering
    at least [pre_bar_open, post_bar_close)) continuously across the
    pre-window bar, every bar in the cluster, and the post-window bar --
    finalizing each forming bar into fixed history as its close boundary
    is crossed, and re-running the pipeline after every single tick to
    record `side`'s signal count at that instant.

    cluster_first_bar_index: positional index (in full_df) of the
    cluster's FIRST flagged bar (i.e. the pre-window bar is index-1).

    Returns a dict:
        {
          "epochs": [...],           # every tick epoch processed
          "counts": [...],           # side's signal count at each tick
          "bar_boundaries": {epoch: "pre"|"cluster"|"post", ...},
          "qualifying": [...],       # bool per tick, count >= any category present
        }
    or None if there isn't enough warmup context / ticks to proceed.
    """
    pre_bar_index = cluster_first_bar_index - 1
    post_bar_index = cluster_first_bar_index + num_cluster_bars  # bar AFTER the cluster's last bar

    if pre_bar_index < 0 or post_bar_index >= len(full_df):
        return ("edge_of_history", f"cluster at edge of candle history "
                f"(pre_idx={pre_bar_index}, post_idx={post_bar_index}, df_len={len(full_df)})")

    warmup = build_warmup_history(full_df, pre_bar_index, warmup_bars)
    if len(warmup) < 40:
        return ("warmup_too_short", "cluster too close to start of candle cache")

    pre_bar_open = int(full_df.iloc[pre_bar_index]["epoch"]) if "epoch" in full_df.columns else None
    # bar boundaries: pre-bar, then each cluster bar, then post-bar
    bar_opens = [cluster_first_bar_index - 1 + i for i in range(num_cluster_bars + 2)]
    # bar_opens are POSITIONAL indices into full_df for [pre, c1, c2, ..., cN, post]

    completed_rows = []  # rows finalized as we cross bar boundaries
    trajectory_epochs, trajectory_counts, trajectory_qualifying = [], [], []
    boundary_label = {}

    # Yield to event loop every N ticks so the websocket keepalive
    # (ping/pong) can run. Without this, a long sync compute loop
    # starves the event loop and the connection drops (WinError 64).
    TICK_YIELD_EVERY = 50
    tick_counter = 0

    labels = ["pre"] + ["cluster"] * num_cluster_bars + ["post"]
    bar_idx_ptr = 0  # which of bar_opens we're currently forming
    current_bar_open_epoch = None
    current_open_price = None
    running_high = running_low = None

    for epoch, price in ticks:
        # advance bar_idx_ptr past any bar whose window this tick has moved beyond
        while bar_idx_ptr < len(bar_opens) - 1:
            this_bar_close_epoch = None
            # bar close epoch = bar open epoch + granularity; determine from full_df position
            pos = bar_opens[bar_idx_ptr]
            bar_open_epoch = int(full_df.iloc[pos]["epoch"])
            this_bar_close_epoch = bar_open_epoch + granularity
            if epoch < this_bar_close_epoch:
                break
            # finalize the current forming bar (if one was started) and move on
            if current_open_price is not None:
                completed_rows.append({
                    "open": current_open_price, "high": running_high,
                    "low": running_low, "close": running_close_val, "volume": 1.0,
                })
            bar_idx_ptr += 1
            current_bar_open_epoch = None
            current_open_price = None

        if bar_idx_ptr >= len(bar_opens):
            break  # ran past the post-bar's own window, nothing more to track

        if current_open_price is None:
            current_bar_open_epoch = int(full_df.iloc[bar_opens[bar_idx_ptr]]["epoch"])
            current_open_price = price
            running_high = price
            running_low = price
        running_high = max(running_high, price)
        running_low = min(running_low, price)
        running_close_val = price

        partial_row = pd.DataFrame([{
            "open": current_open_price, "high": running_high,
            "low": running_low, "close": price, "volume": 1.0,
        }])
        if completed_rows:
            working_df = pd.concat([warmup, pd.DataFrame(completed_rows), partial_row], ignore_index=True)
        else:
            working_df = pd.concat([warmup, partial_row], ignore_index=True)

        try:
            out = run_pipeline(working_df, **pipeline_kwargs)
        except Exception:
            continue

        last = out.iloc[-1]
        count = last["buy_count"] if side == "buy" else last["sell_count"]
        categories = last["buy_categories"] if side == "buy" else last["sell_categories"]

        trajectory_epochs.append(epoch)
        trajectory_counts.append(int(count))
        trajectory_qualifying.append(bool(categories))
        boundary_label[epoch] = labels[bar_idx_ptr]

        tick_counter += 1
        if tick_counter % TICK_YIELD_EVERY == 0:
            await asyncio.sleep(0)

    if not trajectory_epochs:
        return None

    return {
        "epochs": trajectory_epochs,
        "counts": trajectory_counts,
        "qualifying": trajectory_qualifying,
        "bar_label": boundary_label,
    }
