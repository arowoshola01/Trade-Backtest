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
