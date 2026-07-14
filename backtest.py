"""
backtest.py
Orchestrates the full two-pass backtest across every config in
config.BACKTEST_CONFIGS.

Pass 1 (candle resolution): run_pipeline() flags every CALL/PUT bar.
Pass 2 (tick resolution):   for each flagged bar, replay its ticks from
                            candle-open to find the TRUE entry tick/price/
                            category (may differ from Pass 1's bar-close
                            category -- see tick_replay.py).
Settlement:                 for each tested duration, pull ticks out to
                            entry + duration and score win/loss against
                            the entry price (tie = loss).

This makes real network calls against Deriv (deriv_client.py) and cannot
run inside Claude's sandbox -- run it on your own machine:
    python backtest.py
"""

import asyncio
import glob
import os
import time
from collections import defaultdict

import pandas as pd
import websockets.exceptions

import config
from deriv_client import DerivClient, DerivAPIError
from pipeline import run_pipeline
from tick_replay import find_entry_tick, find_settlement_price, iter_bar_ticks
from strategy import StrategyConfig
import checkpoint as ckpt
from email_notifier import (
    load_smtp_env_from_app_password,
    send_email_with_attachments,
    send_final_results_email,
    send_summary_email,
    should_notify,
)


TICK_PACE_DELAY = 0.3  # seconds between per-bar tick pulls, conservative default
CHECKPOINT_EVERY = 10  # save progress every N processed bars
RECONNECT_MAX_ATTEMPTS = 5
RECONNECT_BACKOFF_BASE = 2   # seconds; exponential: 2, 4, 8, 16, 32 -- for dropped connections
RATE_LIMIT_BACKOFF_BASE = 3  # seconds; exponential: 3, 6, 12, 24, 48 -- for Deriv's "RateLimit" error


def format_final_results_for_email(df, row_type: str) -> str:
    if row_type == "combo":
        header = "config | duration | combo | W/L/T | win_rate_pct"
        rows = [
            f"{row['config']} | {int(row['duration_min'])} | {row['combo']} | "
            f"{int(row['wins'])}/{int(row['losses'])}/{int(row['trades'])} | {row['win_rate_pct']:.2f}%"
            for _, row in df.iterrows()
        ]
    else:
        header = "config | duration | category | W/L/T | win_rate_pct"
        rows = [
            f"{row['config']} | {int(row['duration_min'])} | {row['category']} | "
            f"{int(row['wins'])}/{int(row['losses'])}/{int(row['trades'])} | {row['win_rate_pct']:.2f}%"
            for _, row in df.iterrows()
        ]
    return "\n".join([header, "-" * len(header)] + rows)


class BacktestConnectionError(Exception):
    """Raised when a network call fails even after exhausting reconnect attempts."""
    pass


async def call_with_reconnect(client: DerivClient, label: str, description: str,
                               coro_func, *args, max_attempts: int = RECONNECT_MAX_ATTEMPTS, **kwargs):
    """
    Calls coro_func(*args, **kwargs) (an awaitable client method).

    Two distinct failure modes get retried with backoff, up to
    max_attempts each:
      - Dropped connection (ConnectionClosed / OSError): waits, then
        reconnects the client's WebSocket before retrying.
      - Deriv's "RateLimit" API error: waits (no reconnect needed, the
        socket itself is fine, just the request rate), then retries the
        same request.

    Any OTHER DerivAPIError (bad params, no data for that range, etc.) is
    NOT retried -- retrying wouldn't fix it, so it propagates immediately
    for the caller to treat as a genuine per-bar outcome.
    """
    last_exc = None
    for attempt in range(max_attempts + 1):
        try:
            return await coro_func(*args, **kwargs)
        except (websockets.exceptions.ConnectionClosed, OSError) as e:
            last_exc = e
            if attempt < max_attempts:
                wait = RECONNECT_BACKOFF_BASE * (2 ** attempt)
                print(f"\n[{label}] {description}: connection issue ({e}), "
                      f"reconnecting (attempt {attempt + 1}/{max_attempts}) in {wait}s...")
                await asyncio.sleep(wait)
                try:
                    await client.reconnect()
                except Exception as reconnect_exc:
                    print(f"[{label}] reconnect attempt failed: {reconnect_exc}")
            else:
                raise BacktestConnectionError(
                    f"{label}: {description} failed after {max_attempts} reconnect attempts") from e
        except DerivAPIError as e:
            if e.code != "RateLimit":
                raise  # genuine API-level rejection -- not retryable, let the caller handle it
            last_exc = e
            if attempt < max_attempts:
                wait = RATE_LIMIT_BACKOFF_BASE * (2 ** attempt)
                print(f"\n[{label}] {description}: rate limited, waiting {wait}s "
                      f"(attempt {attempt + 1}/{max_attempts})...")
                await asyncio.sleep(wait)
            else:
                raise BacktestConnectionError(
                    f"{label}: {description} failed after {max_attempts} rate-limit retries") from e
    raise BacktestConnectionError(f"{label}: {description} failed") from last_exc


def make_combo_label(categories: list) -> str:
    cats = tuple(sorted(categories))
    if cats == ("direct",):
        return "direct only"
    if cats == ("fallback",):
        return "fallback only"
    if cats == ("regime_confirmation",):
        return "regime_confirmation only"
    if cats == ("direct", "regime_confirmation"):
        return "direct + regime_confirmation"
    if cats == ("fallback", "regime_confirmation"):
        return "fallback + regime_confirmation"
    # direct+fallback / direct+fallback+regime_confirmation are structurally
    # impossible (see strategy.py docstring) -- if this ever appears it
    # means the two-pass logic disagreed with itself; label it visibly
    # rather than silently mis-bucketing.
    return "UNEXPECTED: " + "+".join(cats) if cats else "NONE"


async def pull_candles(client: DerivClient, granularity: int, weeks: float) -> pd.DataFrame:
    if config.BACKTEST_HISTORY_CANDLES is not None:
        total_count = config.BACKTEST_HISTORY_CANDLES
    else:
        total_count = int(weeks * 7 * 24 * 3600 / granularity)
    candles = await client.get_candle_history_paginated(granularity=granularity, total_count=total_count)
    df = pd.DataFrame(candles)
    df = df.sort_values("epoch").reset_index(drop=True)
    return df


def drop_edge_of_data_bars(flagged: pd.DataFrame, df: pd.DataFrame, granularity: int, max_duration_sec: int):
    if not config.DROP_EDGE_OF_DATA_SIGNALS or len(df) == 0:
        return flagged
    last_epoch = int(df["epoch"].iloc[-1])
    cutoff = last_epoch + granularity - max_duration_sec  # bar open must be at/before this to fully resolve
    keep_mask = flagged.index.map(lambda bi: int(df["epoch"].iloc[bi]) <= cutoff)
    dropped = (~pd.Series(list(keep_mask), index=flagged.index)).sum()
    if dropped:
        print(f"  Dropped {dropped} edge-of-data bar(s) that couldn't be fully resolved.")
    return flagged[list(keep_mask)]


def gather_checkpoint_summary():
    summary_items = []
    for label, _, _ in config.BACKTEST_CONFIGS:
        state = ckpt.load_checkpoint(label)
        if state is None:
            continue
        processed_count = len(state.get("processed_bar_epochs", []))
        total_count = len(state.get("flagged_bar_epochs", []))
        trades_count = len(state.get("trades", []))
        summary_items.append((label, processed_count, total_count, trades_count))
    return summary_items


async def run_config_backtest(client: DerivClient, label: str, chart_timeframe: str, tf_cal_override):
    granularity, tf_cal = config.resolve_granularity_and_tf_cal(chart_timeframe, tf_cal_override)
    durations_min = config.BACKTEST_DURATIONS_MIN[chart_timeframe]
    max_duration_sec = max(durations_min) * 60

    # ── candle pull, cached ──
    df = ckpt.load_candles_cache(label)
    if df is not None:
        print(f"[{label}] Loaded {len(df)} candles from cache.")
    else:
        if config.BACKTEST_HISTORY_CANDLES is not None:
            print(f"[{label}] Pulling {config.BACKTEST_HISTORY_CANDLES} {chart_timeframe} candles "
                  f"(tf_cal={tf_cal})...")
        else:
            print(f"[{label}] Pulling {config.BACKTEST_HISTORY_WEEKS} weeks of {chart_timeframe} candles "
                  f"(tf_cal={tf_cal})...")
        df = await call_with_reconnect(client, label, "candle pull",
                                        pull_candles, client, granularity, config.BACKTEST_HISTORY_WEEKS)
        print(f"[{label}] Pulled {len(df)} candles.")
        ckpt.save_candles_cache(label, df)

    # epoch <-> positional-index lookup for THIS session's DataFrame.
    # Bar identity persisted to disk is always the epoch (see checkpoint.py);
    # positions are only ever used internally to index into `out`/`df`.
    epoch_to_pos = {int(e): i for i, e in enumerate(df["epoch"].values)}

    pipeline_kwargs = dict(
        tf_cal=tf_cal, movol_model=config.MOVOL_MODEL, seconds_per_bar=granularity,
        movol_lookback_hours=config.MOVOL_LOOKBACK_HOURS,
        strategy_cfg=StrategyConfig(threshold=config.SIGNAL_THRESHOLD),
    )

    print(f"[{label}] Running Pass 1 (bar-close pipeline)...")
    out = run_pipeline(df.drop(columns=["epoch"]), **pipeline_kwargs)
    out["epoch"] = df["epoch"].values

    replay_pipeline_kwargs = dict(
        tf_cal=tf_cal, movol_model=config.MOVOL_MODEL, seconds_per_bar=granularity,
        movol_lookback_hours=config.MOVOL_LOOKBACK_HOURS,
        strategy_cfg=StrategyConfig(threshold=config.SIGNAL_THRESHOLD),
    )

    # ── resume or start fresh ──
    state = ckpt.load_checkpoint(label)
    if state is not None:
        flagged_bar_epochs = state["flagged_bar_epochs"]
        processed_epoch_set = set(state["processed_bar_epochs"])
        skipped_bar_epochs = set(state.get("skipped_bar_epochs", []))
        dropped_bar_epochs = set(state.get("dropped_bar_epochs", []))
        trades = state["trades"]

        completed_trade_epochs = {int(t["bar_epoch"]) for t in trades}
        legacy_retryable_epochs = {int(e) for e in processed_epoch_set if int(e) not in completed_trade_epochs and int(e) not in dropped_bar_epochs}
        retryable_epochs = skipped_bar_epochs | legacy_retryable_epochs
        processed_epoch_set = processed_epoch_set - retryable_epochs
        if retryable_epochs:
            print(f"[{label}] Requeueing {len(retryable_epochs)} previously incomplete bar(s) for retry.")
        print(f"[{label}] Resuming from checkpoint: {len(processed_epoch_set)}/{len(flagged_bar_epochs)} "
              f"bar(s) already processed, {len(trades)} trade(s) so far.")
    else:
        flagged = out[out["decision"].notna()].copy()
        print(f"[{label}] Pass 1 flagged {len(flagged)} bar(s).")
        flagged = drop_edge_of_data_bars(flagged, df, granularity, max_duration_sec)
        print(f"[{label}] {len(flagged)} bar(s) remain for Pass 2.")
        flagged_bar_epochs = [int(out.loc[bi, "epoch"]) for bi in flagged.index]
        processed_epoch_set = set()
        skipped_bar_epochs = set()
        dropped_bar_epochs = set()
        trades = []
        ckpt.save_checkpoint(label, flagged_bar_epochs, list(processed_epoch_set), trades,
                             skipped_bar_epochs=list(skipped_bar_epochs),
                             dropped_bar_epochs=list(dropped_bar_epochs))

    remaining_epochs = [e for e in flagged_bar_epochs if e not in processed_epoch_set]
    total = len(flagged_bar_epochs)
    initial_processed_count = len(processed_epoch_set)

    for n, bar_epoch in enumerate(remaining_epochs, start=1):
        bi = epoch_to_pos.get(bar_epoch)
        if bi is None:
            # This epoch existed in a prior candle pull but isn't present
            # in the currently cached/pulled set (cache was cleared or
            # regenerated with a different window). Can't replay it
            # without the surrounding warmup context -- skip and move on
            # rather than silently mis-mapping it to the wrong bar.
            print(f"[{label}] bar epoch {bar_epoch} not found in current candle set, dropping.")
            processed_epoch_set.add(bar_epoch)
            dropped_bar_epochs.add(bar_epoch)
            skipped_bar_epochs.discard(bar_epoch)
            continue

        row = out.loc[bi]
        side = "buy" if row["decision"] == "CALL" else "sell"
        bar_open_epoch = int(row["epoch"])
        bar_close_epoch = bar_open_epoch + granularity
        settlement_buffer = max_duration_sec + 30

        done_count = initial_processed_count + n
        print(f"[{label}] ({done_count}/{total}) bar@{bar_open_epoch} pulling ticks...", end=" ", flush=True)
        try:
            ticks = await call_with_reconnect(
                client, label, f"tick pull bar@{bar_open_epoch}",
                client.get_tick_history, count=2000, start=bar_open_epoch, end=bar_close_epoch + settlement_buffer)
        except BacktestConnectionError:
            # Connection could not be restored even after retries. Do NOT
            # mark this bar as processed -- it was never actually
            # evaluated. Save what's genuinely done and stop this config;
            # re-running will resume from exactly this bar.
            print(f"\n[{label}] Connection could not be restored after {RECONNECT_MAX_ATTEMPTS} attempts. "
                  f"Stopping this config at bar@{bar_open_epoch} -- re-run to resume.")
            ckpt.save_checkpoint(label, flagged_bar_epochs, list(processed_epoch_set), trades,
                                 skipped_bar_epochs=list(skipped_bar_epochs),
                                 dropped_bar_epochs=list(dropped_bar_epochs))
            raise
        except Exception as e:
            # A genuine, non-connection failure for this specific bar is treated
            # as an incomplete attempt rather than a completed outcome. Keep it
            # unprocessed so a resumed run can retry it; only true "drop"
            # cases (such as no entry tick found) are marked as processed.
            print(f"tick pull failed ({e}), skipping.")
            skipped_bar_epochs.add(bar_epoch)
            if n % CHECKPOINT_EVERY == 0:
                ckpt.save_checkpoint(label, flagged_bar_epochs, list(processed_epoch_set), trades,
                                     skipped_bar_epochs=list(skipped_bar_epochs),
                                     dropped_bar_epochs=list(dropped_bar_epochs))
            continue
        await asyncio.sleep(TICK_PACE_DELAY)

        ticks_in_bar = list(iter_bar_ticks(
            [{"epoch": t["epoch"], "price": t["price"]} for t in ticks], bar_open_epoch, bar_close_epoch))
        entry = find_entry_tick(out, bi, ticks_in_bar, side, replay_pipeline_kwargs)
        if entry is None:
            print("no entry tick found, dropping.")
            processed_epoch_set.add(bar_epoch)
            dropped_bar_epochs.add(bar_epoch)
            skipped_bar_epochs.discard(bar_epoch)
            if n % CHECKPOINT_EVERY == 0:
                ckpt.save_checkpoint(label, flagged_bar_epochs, list(processed_epoch_set), trades,
                                     skipped_bar_epochs=list(skipped_bar_epochs),
                                     dropped_bar_epochs=list(dropped_bar_epochs))
            continue

        ticks_after_entry = [t for t in ticks if t["epoch"] >= entry["epoch"]]
        outcomes = {}
        for dur_min in durations_min:
            settle_price, settle_epoch = find_settlement_price(
                [{"epoch": t["epoch"], "price": t["price"]} for t in ticks_after_entry],
                entry["epoch"], dur_min * 60)
            if settle_price is None:
                continue
            if side == "buy":
                win = settle_price > entry["price"]
            else:
                win = settle_price < entry["price"]
            if settle_price == entry["price"] and config.TIE_COUNTS_AS_LOSS:
                win = False
            outcomes[dur_min] = win

        print(f"entry@{entry['epoch']} cats={entry['categories']} -> {len(outcomes)}/{len(durations_min)} durations scored.")

        trades.append({
            "bar_epoch": bar_epoch, "side": side, "entry_epoch": entry["epoch"],
            "entry_price": entry["price"], "categories": entry["categories"],
            "combo": make_combo_label(entry["categories"]), "outcomes": outcomes,
        })
        processed_epoch_set.add(bar_epoch)
        skipped_bar_epochs.discard(bar_epoch)
        dropped_bar_epochs.discard(bar_epoch)

        if n % CHECKPOINT_EVERY == 0:
            ckpt.save_checkpoint(label, flagged_bar_epochs, list(processed_epoch_set), trades,
                                 skipped_bar_epochs=list(skipped_bar_epochs),
                                 dropped_bar_epochs=list(dropped_bar_epochs))
            print(f"[{label}]   (checkpoint saved: {len(processed_epoch_set)}/{total} processed, {len(trades)} trades)")
            if should_notify(len(processed_epoch_set), 50):
                summary_items = gather_checkpoint_summary()
                if summary_items:
                    send_summary_email(summary_items, event="checkpoint")

    # final save, covers any tail not divisible by CHECKPOINT_EVERY
    ckpt.save_checkpoint(label, flagged_bar_epochs, list(processed_epoch_set), trades,
                         skipped_bar_epochs=list(skipped_bar_epochs),
                         dropped_bar_epochs=list(dropped_bar_epochs))

    return trades, durations_min


def build_results_tables(all_results: dict):
    """
    all_results: {config_label: (trades, durations_min)}
    Returns (combo_table_df, per_category_table_df) -- both long-format,
    one row per (config, duration, bucket).
    """
    combo_rows = []
    category_rows = []

    for label, (trades, durations_min) in all_results.items():
        for dur_min in durations_min:
            combo_stats = defaultdict(lambda: [0, 0])   # combo -> [wins, losses]
            cat_stats = defaultdict(lambda: [0, 0])      # category -> [wins, losses] (inclusive, can overlap)

            for trade in trades:
                if dur_min not in trade["outcomes"]:
                    continue
                win = trade["outcomes"][dur_min]
                combo_stats[trade["combo"]][0 if win else 1] += 1
                for cat in trade["categories"]:
                    cat_stats[cat][0 if win else 1] += 1

            for combo, (w, l) in combo_stats.items():
                total = w + l
                combo_rows.append({
                    "config": label, "duration_min": dur_min, "combo": combo,
                    "wins": w, "losses": l, "trades": total,
                    "win_rate_pct": round(100 * w / total, 2) if total else None,
                })
            for cat, (w, l) in cat_stats.items():
                total = w + l
                category_rows.append({
                    "config": label, "duration_min": dur_min, "category": cat,
                    "wins": w, "losses": l, "trades": total,
                    "win_rate_pct": round(100 * w / total, 2) if total else None,
                })

    combo_df = pd.DataFrame(combo_rows)
    category_df = pd.DataFrame(category_rows)
    return combo_df, category_df


async def main():
    load_smtp_env_from_app_password()

    if is_configured():
        print("SMTP email notifications are enabled.")
    else:
        print("SMTP email notifications are disabled or incomplete; backtest will still run.")

    client = DerivClient()
    await client.connect()
    auth = await client.authorize()
    print(f"Authorized as {auth.get('loginid')} (is_virtual={auth.get('is_virtual')})\n")

    all_results = {}
    try:
        for label, chart_timeframe, tf_cal_override in config.BACKTEST_CONFIGS:
            print(f"\n===== Config: {label} =====")
            t0 = time.time()
            trades, durations_min = await run_config_backtest(client, label, chart_timeframe, tf_cal_override)
            all_results[label] = (trades, durations_min)
            print(f"[{label}] Done: {len(trades)} scorable trade(s) in {time.time()-t0:.1f}s.")
    except KeyboardInterrupt:
        print("\n\nInterrupted -- progress up to the last checkpoint (every "
              f"{CHECKPOINT_EVERY} bars) has been saved. Just re-run "
              "`python backtest.py` to resume from where this left off.")
        await client.close()
        return
    except BacktestConnectionError as e:
        print(f"\n\n{e}\n"
              "Progress up to the point of failure has been saved. Just re-run "
              "`python backtest.py` to resume -- it will pick up exactly where this left off.")
        await client.close()
        return
    finally:
        try:
            await client.close()
        except Exception:
            pass

    combo_df, category_df = build_results_tables(all_results)

    summary_items = []
    for label in all_results:
        state = ckpt.load_checkpoint(label)
        if state is None:
            continue
        processed_count = len(state.get("processed_bar_epochs", []))
        total_count = len(state.get("flagged_bar_epochs", []))
        trades_count = len(state.get("trades", []))
        summary_items.append((label, processed_count, total_count, trades_count))
    if summary_items:
        send_summary_email(summary_items, event="done")

    combo_df.to_csv("backtest_combo_results.csv", index=False)
    category_df.to_csv("backtest_category_results.csv", index=False)

    combo_text = format_final_results_for_email(combo_df, "combo")
    category_text = format_final_results_for_email(category_df, "category")
    send_final_results_email(combo_text, category_text)

    checkpoint_csvs = sorted(glob.glob("backtest_checkpoints/*.csv"))
    checkpoint_jsons = sorted(glob.glob("backtest_checkpoints/*.json"))
    result_csvs = ["backtest_combo_results.csv", "backtest_category_results.csv"]
    attachment_paths = checkpoint_csvs + checkpoint_jsons + result_csvs
    if attachment_paths:
        attachment_list = "\n".join(f"- {path}" for path in attachment_paths)
        send_email_with_attachments(
            "[Backtest] backtest report attachments",
            "Attached are the backtest checkpoint and result files:\n\n" + attachment_list,
            attachment_paths,
        )

    print("\n\n===== COMBO RESULTS (mutually exclusive, sums to total trades) =====")
    print(combo_text)
    print("\n\n===== PER-CATEGORY RESULTS (inclusive, overlaps counted in each) =====")
    print(category_text)
    print("\nSaved: backtest_combo_results.csv, backtest_category_results.csv")


if __name__ == "__main__":
    asyncio.run(main())
