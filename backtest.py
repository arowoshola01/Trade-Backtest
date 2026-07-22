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
from tick_replay import find_entry_tick, find_settlement_price, iter_bar_ticks, cluster_flagged_bars
from strategy import StrategyConfig
import checkpoint as ckpt
from network_retry import (call_with_reconnect, BacktestConnectionError, TICK_PACE_DELAY,
                            RECONNECT_MAX_ATTEMPTS, RECONNECT_BACKOFF_BASE, RATE_LIMIT_BACKOFF_BASE)
from cluster_trajectory import capture_cluster_trajectories
import repaint_analysis
from email_notifier import (
    format_table,
    is_configured,
    load_smtp_env_from_app_password,
    missing_smtp_env_vars,
    send_email,
    send_email_with_attachments,
    send_final_results_email,
    send_raw_email,
    send_summary_email,
    should_notify,
)


CHECKPOINT_EVERY = 10  # save progress every N processed bars


def format_final_results_for_email(df, row_type: str) -> str:
    if row_type == "combo":
        headers = ["config", "duration", "combo", "W/L/T", "win_rate_pct"]
        rows = [
            [
                str(row["config"]),
                str(int(row["duration_min"])),
                str(row["combo"]),
                f"{int(row['wins'])}/{int(row['losses'])}/{int(row['trades'])}",
                f"{row['win_rate_pct']:.2f}%" if row["win_rate_pct"] is not None else "N/A",
            ]
            for _, row in df.iterrows()
        ]
    else:
        headers = ["config", "duration", "category", "W/L/T", "win_rate_pct"]
        rows = [
            [
                str(row["config"]),
                str(int(row["duration_min"])),
                str(row["category"]),
                f"{int(row['wins'])}/{int(row['losses'])}/{int(row['trades'])}",
                f"{row['win_rate_pct']:.2f}%" if row["win_rate_pct"] is not None else "N/A",
            ]
            for _, row in df.iterrows()
        ]
    return format_table(rows, headers)


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


async def run_config_backtest(client: DerivClient, label: str, chart_timeframe: str, tf_cal_override):
    granularity, tf_cal = config.resolve_granularity_and_tf_cal(chart_timeframe, tf_cal_override)
    durations_min = config.BACKTEST_DURATIONS_MIN[chart_timeframe]
    max_duration_sec = max(durations_min) * 60

    # ── candle pull, cached ──
    df = ckpt.load_candles_cache(label)
    if df is not None:
        print(f"[{label}] {config.BACKTEST_HISTORY_CANDLES} candle pulled.")
        print(f"[{label}] Loading from cache...")
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
        print(f"[{label}] {len(df)} candle pulled.")
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
        prior_clusters = len(state.get("cluster_trajectories", {}))
        # Compute the TOTAL cluster count (captured + remaining) so the
        # "From checkpoint" line reflects the full cluster set, not just
        # the trajectories captured so far. cluster_flagged_bars groups
        # flagged bars into same-direction consecutive-bar clusters --
        # same grouping used inside capture_cluster_trajectories().
        epoch_side_pairs = []
        for e in flagged_bar_epochs:
            pos = epoch_to_pos.get(e)
            if pos is None:
                continue
            row = out.iloc[pos]
            epoch_side_pairs.append((e, "buy" if row["decision"] == "CALL" else "sell"))
        total_clusters = len(cluster_flagged_bars(epoch_side_pairs, granularity))
        print(f"[{label}] From checkpoint: {len(flagged_bar_epochs)} bar(s) flagged, "
              f"{total_clusters} cluster captured, {len(processed_epoch_set)}/{len(flagged_bar_epochs)} "
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
                             dropped_bar_epochs=list(dropped_bar_epochs),
                             skipped_cluster_ids=[])

    existing_trajectory_ids = set(state.get("cluster_trajectories", {}).keys()) if state is not None else set()
    cluster_trajectories, epoch_to_cluster_id, skipped_cluster_ids = await capture_cluster_trajectories(
        client, label, out, df, flagged_bar_epochs, existing_trajectory_ids, epoch_to_pos,
        granularity, max_duration_sec, replay_pipeline_kwargs,
        checkpoint_state=state,
        processed_epoch_set=processed_epoch_set,
        trades=trades,
        skipped_bar_epochs=skipped_bar_epochs,
        dropped_bar_epochs=dropped_bar_epochs)
    if state is not None:
        # preserve any trajectories captured in a prior (interrupted) run
        existing_trajectories = state.get("cluster_trajectories", {})
        cluster_trajectories = {**existing_trajectories, **cluster_trajectories}

    if skipped_cluster_ids:
        print(f"[{label}] {len(skipped_cluster_ids)} cluster(s) still at edge of history, will retry on next resume.")

    # retroactively link any already-existing trades (from a prior partial
    # run) to their cluster, now that we have a full epoch->cluster_id map
    for t in trades:
        be = int(t["bar_epoch"])
        if be in epoch_to_cluster_id and t.get("cluster_id") != epoch_to_cluster_id[be]:
            t["cluster_id"] = epoch_to_cluster_id[be]

    if cluster_trajectories or any(t.get("cluster_id") for t in trades):
        ckpt.save_checkpoint(label, flagged_bar_epochs, list(processed_epoch_set), trades,
                             skipped_bar_epochs=list(skipped_bar_epochs),
                             dropped_bar_epochs=list(dropped_bar_epochs),
                             cluster_trajectories=cluster_trajectories,
                             skipped_cluster_ids=list(skipped_cluster_ids))

    remaining_epochs = [e for e in flagged_bar_epochs if e not in processed_epoch_set]
    total = len(flagged_bar_epochs)
    initial_processed_count = len(processed_epoch_set)

    print(f"[{label}] Running Pass 2 (replay pipeline)...")

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
        print(f"[{label}] ({done_count}/{total}) pulling ticks...", end=" ", flush=True)

        # Try to reuse ticks from cluster capture if this bar belongs to a
        # cluster and we have cached ticks that cover at least part of this
        # bar's window. Three cases:
        #   1. Cached window fully covers needed window → reuse only
        #   2. Cached window partially covers → reuse cached + pull missing
        #   3. No cached ticks or no overlap → pull fresh
        ticks = None
        reuse_mode = None  # "full", "partial", or None
        cluster_id = epoch_to_cluster_id.get(bar_epoch)
        needed_start = bar_open_epoch
        needed_end = bar_close_epoch + settlement_buffer
        if cluster_id is not None and cluster_id in cluster_trajectories:
            traj = cluster_trajectories[cluster_id]
            cached_ticks = traj.get("ticks")
            window_start = traj.get("window_start")
            window_end = traj.get("window_end")
            if cached_ticks is not None and window_start is not None and window_end is not None:
                if window_start <= needed_start and window_end >= needed_end:
                    # Full coverage: cached window contains the needed window
                    ticks = [t for t in cached_ticks if needed_start <= t["epoch"] <= needed_end]
                    reuse_mode = "full"
                elif window_end > needed_start and window_start < needed_end:
                    # Partial overlap: use what's cached, pull the rest
                    reuse_mode = "partial"
                    # Use cached ticks that fall within the needed window
                    cached_part = [t for t in cached_ticks if needed_start <= t["epoch"] <= needed_end]
                    # Determine which sub-ranges are missing
                    missing_ranges = []
                    if window_start > needed_start:
                        missing_ranges.append((needed_start, window_start))
                    if window_end < needed_end:
                        missing_ranges.append((window_end, needed_end))
                    # Pull each missing range and merge
                    pulled_parts = []
                    partial_failed = False
                    for ms, me in missing_ranges:
                        try:
                            part = await call_with_reconnect(
                                client, label, f"tick pull bar@{bar_open_epoch} gap {ms}-{me}",
                                client.get_tick_history, count=2000, start=ms, end=me)
                            pulled_parts.extend(part)
                        except BacktestConnectionError:
                            # Connection death during partial pull — same
                            # save-and-stop logic as the full-pull case.
                            print(f"\n[{label}] Connection could not be restored after {RECONNECT_MAX_ATTEMPTS} attempts. "
                                  f"Stopping this config at bar@{bar_open_epoch} -- re-run to resume.")
                            ckpt.save_checkpoint(label, flagged_bar_epochs, list(processed_epoch_set), trades,
                                                 skipped_bar_epochs=list(skipped_bar_epochs),
                                                 dropped_bar_epochs=list(dropped_bar_epochs),
                                                 cluster_trajectories=cluster_trajectories,
                                                 skipped_cluster_ids=list(skipped_cluster_ids))
                            raise
                        except Exception as e:
                            print(f"partial tick pull failed ({e}), skipping.")
                            skipped_bar_epochs.add(bar_epoch)
                            partial_failed = True
                            break
                    if partial_failed:
                        if n % CHECKPOINT_EVERY == 0:
                            ckpt.save_checkpoint(label, flagged_bar_epochs, list(processed_epoch_set), trades,
                                                 skipped_bar_epochs=list(skipped_bar_epochs),
                                                 dropped_bar_epochs=list(dropped_bar_epochs),
                                                 cluster_trajectories=cluster_trajectories,
                                                 skipped_cluster_ids=list(skipped_cluster_ids))
                        continue
                    # Merge cached + pulled, sort by epoch
                    ticks = cached_part + pulled_parts
                    ticks.sort(key=lambda t: t["epoch"])

        if reuse_mode == "full":
            print(f"reused cached ticks", end=" ", flush=True)
        elif reuse_mode == "partial":
            print(f"reusing cached ticks & pulling other ticks...", end=" ", flush=True)

        if ticks is None:
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
                                     dropped_bar_epochs=list(dropped_bar_epochs),
                                     cluster_trajectories=cluster_trajectories,
                                     skipped_cluster_ids=list(skipped_cluster_ids))
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
                                         dropped_bar_epochs=list(dropped_bar_epochs),
                                         cluster_trajectories=cluster_trajectories,
                                         skipped_cluster_ids=list(skipped_cluster_ids))
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
                                     dropped_bar_epochs=list(dropped_bar_epochs),
                                     cluster_trajectories=cluster_trajectories,
                                     skipped_cluster_ids=list(skipped_cluster_ids))
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
            "cluster_id": epoch_to_cluster_id.get(bar_epoch),
        })
        processed_epoch_set.add(bar_epoch)
        skipped_bar_epochs.discard(bar_epoch)
        dropped_bar_epochs.discard(bar_epoch)

        if n % CHECKPOINT_EVERY == 0:
            ckpt.save_checkpoint(label, flagged_bar_epochs, list(processed_epoch_set), trades,
                                 skipped_bar_epochs=list(skipped_bar_epochs),
                                 dropped_bar_epochs=list(dropped_bar_epochs),
                                 cluster_trajectories=cluster_trajectories,
                                 skipped_cluster_ids=list(skipped_cluster_ids))
            print(f"[{label}]   (checkpoint saved: {len(processed_epoch_set)}/{total} processed, {len(trades)} trades)")

        # Strict every-50-processed-bars notification, independent of the
        # CHECKPOINT_EVERY cadence. Uses the processed count (not the
        # iteration counter n) so skipped/dropped bars don't shift the
        # notification boundary.
        if should_notify(len(processed_epoch_set), 50):
            summary_items = [(label, len(processed_epoch_set), total, len(trades))]
            if summary_items:
                send_summary_email(summary_items, event="checkpoint")

    # final save, covers any tail not divisible by CHECKPOINT_EVERY
    ckpt.save_checkpoint(label, flagged_bar_epochs, list(processed_epoch_set), trades, skipped_bar_epochs=list(skipped_bar_epochs), dropped_bar_epochs=list(dropped_bar_epochs), cluster_trajectories=cluster_trajectories, skipped_cluster_ids=list(skipped_cluster_ids))
    return trades, durations_min, len(processed_epoch_set)


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


RECONCILE_ZONE_BARS = 300
# Matches tick_replay.WARMUP_BARS. Bars within this many positions of the
# OLD dataset's original start are the only ones whose Pass 1
# classification could change when older history is prepended -- Pass 2
# never looks further back than this per bar anyway, and the HMM/rolling
# indicators converge well within this span. Everything beyond it is
# provably unaffected by extending history further backward.


async def extend_config_candles(client: DerivClient, label: str, chart_timeframe: str, tf_cal_override,
                                  target_count: int):
    """
    Extends an existing candle cache backward to target_count candles,
    reruns Pass 1 on the enlarged set, and reconciles the bounded
    boundary zone (see RECONCILE_ZONE_BARS) against the existing Pass 2
    checkpoint -- any bar whose classification changed gets un-marked and
    requeued for a fresh Pass 2 pass; everything else is left untouched.
    Does NOT run Pass 2 itself -- call run_config_backtest afterward (or
    let main()'s normal loop do it) to process whatever's newly pending.
    """
    granularity, tf_cal = config.resolve_granularity_and_tf_cal(chart_timeframe, tf_cal_override)
    durations_min = config.BACKTEST_DURATIONS_MIN[chart_timeframe]
    max_duration_sec = max(durations_min) * 60

    df_old = ckpt.load_candles_cache(label)
    if df_old is None:
        print(f"[{label}] No existing candle cache -- nothing to extend. Run a normal backtest first.")
        return
    df_old = df_old.sort_values("epoch").reset_index(drop=True)

    current_count = len(df_old)
    if current_count >= target_count:
        print(f"[{label}] Already have {current_count} candles (>= target {target_count}). Nothing to extend.")
        return

    additional_needed = target_count - current_count
    earliest_epoch = int(df_old["epoch"].iloc[0])
    print(f"[{label}] Extending: have {current_count}, pulling {additional_needed} older candle(s) "
          f"(target {target_count})...")

    older = await call_with_reconnect(
        client, label, "extend candle pull",
        client.get_candle_history_paginated, granularity=granularity,
        total_count=additional_needed, end=earliest_epoch - 1)
    print(f"[{label}] Pulled {len(older)} older candle(s).")
    if not older:
        print(f"[{label}] No older history available from Deriv -- may have hit the retention limit.")
        return

    df_older = pd.DataFrame(older)
    df_merged = pd.concat([df_older, df_old], ignore_index=True)
    df_merged = df_merged.drop_duplicates(subset="epoch").sort_values("epoch").reset_index(drop=True)
    ckpt.save_candles_cache(label, df_merged)
    print(f"[{label}] Merged cache: {len(df_merged)} candles total "
          f"(+{len(df_merged) - current_count} new).")

    # ── recompute Pass 1 on the full merged set ──
    pipeline_kwargs = dict(
        tf_cal=tf_cal, movol_model=config.MOVOL_MODEL, seconds_per_bar=granularity,
        movol_lookback_hours=config.MOVOL_LOOKBACK_HOURS,
        strategy_cfg=StrategyConfig(threshold=config.SIGNAL_THRESHOLD),
    )
    out_new = run_pipeline(df_merged.drop(columns=["epoch"]), **pipeline_kwargs)
    out_new["epoch"] = df_merged["epoch"].values

    flagged_new = out_new[out_new["decision"].notna()].copy()
    flagged_new = drop_edge_of_data_bars(flagged_new, df_merged, granularity, max_duration_sec)
    new_flagged_epochs = {int(out_new.loc[bi, "epoch"]) for bi in flagged_new.index}
    new_side_by_epoch = {
        int(out_new.loc[bi, "epoch"]): ("buy" if out_new.loc[bi, "decision"] == "CALL" else "sell")
        for bi in flagged_new.index
    }

    # ── load existing Pass 2 progress ──
    state = ckpt.load_checkpoint(label)
    if state is None:
        old_flagged = set()
        processed_epoch_set, skipped_bar_epochs, dropped_bar_epochs = set(), set(), set()
        trades = []
        cluster_trajectories = {}
        skipped_cluster_ids = set()
    else:
        old_flagged = {int(e) for e in state["flagged_bar_epochs"]}
        processed_epoch_set = {int(e) for e in state["processed_bar_epochs"]}
        skipped_bar_epochs = {int(e) for e in state.get("skipped_bar_epochs", [])}
        dropped_bar_epochs = {int(e) for e in state.get("dropped_bar_epochs", [])}
        trades = state["trades"]
        cluster_trajectories = state.get("cluster_trajectories", {})
        skipped_cluster_ids = set(state.get("skipped_cluster_ids", []))
        for t in trades:
            t["outcomes"] = {int(k): v for k, v in t["outcomes"].items()}

    # ── reconcile only the bounded boundary zone ──
    boundary_epochs = {int(e) for e in df_old["epoch"].iloc[:RECONCILE_ZONE_BARS].values}
    reconciled_away = set()

    for epoch in boundary_epochs & old_flagged & processed_epoch_set:
        old_trade = next((t for t in trades if int(t["bar_epoch"]) == epoch), None)
        still_flagged = epoch in new_flagged_epochs
        if not still_flagged:
            changed = True
        elif old_trade is not None:
            changed = new_side_by_epoch.get(epoch) != old_trade["side"]
        else:
            # It was a permanent "no entry tick found" drop, which never
            # recorded which side it was evaluating. Can't positively
            # confirm the side is unchanged, so be conservative and
            # requeue it rather than risk silently keeping a stale drop.
            changed = True
        if changed:
            reconciled_away.add(epoch)

    if reconciled_away:
        print(f"[{label}] Reconciliation: {len(reconciled_away)} boundary bar(s) changed classification "
              f"under the extended history -- requeuing for fresh Pass 2.")
        trades = [t for t in trades if int(t["bar_epoch"]) not in reconciled_away]
        processed_epoch_set -= reconciled_away
        skipped_bar_epochs -= reconciled_away
        dropped_bar_epochs -= reconciled_away
    else:
        print(f"[{label}] Reconciliation: no boundary bars changed classification.")

    flagged_bar_epochs = sorted(new_flagged_epochs)
    ckpt.save_checkpoint(label, flagged_bar_epochs, list(processed_epoch_set), trades,
                         skipped_bar_epochs=list(skipped_bar_epochs),
                         dropped_bar_epochs=list(dropped_bar_epochs),
                         cluster_trajectories=cluster_trajectories,
                         skipped_cluster_ids=list(skipped_cluster_ids))
    pending = len(flagged_bar_epochs) - len(processed_epoch_set)
    print(f"[{label}] Extend complete: {len(flagged_bar_epochs)} total flagged bar(s), "
          f"{len(processed_epoch_set)} already resolved, {pending} pending for Pass 2.\n")


async def main():
    import sys
    do_extend = "--extend" in sys.argv

    load_smtp_env_from_app_password()

    if is_configured():
        print("SMTP email notifications are enabled.")
    else:
        print(f"SMTP email notifications are disabled; missing env vars: {missing_smtp_env_vars()}. Backtest will still run.")

    client = DerivClient()
    await client.connect()
    auth = await client.authorize()
    print(f"Authorized as {auth.get('loginid')} (is_virtual={auth.get('is_virtual')})")
    print(f"Market Symbol: {config.SYMBOL}\n")

    if do_extend:
        target = config.BACKTEST_HISTORY_CANDLES
        if target is None:
            print("--extend requires config.BACKTEST_HISTORY_CANDLES to be set to a target count. Aborting.")
            await client.close()
            return
        print(f"===== EXTEND MODE: target {target} candles per config =====\n")
        try:
            for label, chart_timeframe, tf_cal_override in config.BACKTEST_CONFIGS:
                await extend_config_candles(client, label, chart_timeframe, tf_cal_override, target)
        except (KeyboardInterrupt, BacktestConnectionError) as e:
            print(f"\n\nExtend step interrupted: {e}\n"
                  "Any config that finished extending has already been merged and reconciled on disk. "
                  "Just re-run `python backtest.py --extend` to continue with any remaining configs.")
            await client.close()
            return
        print("===== EXTEND MODE complete -- proceeding to normal Pass 2 processing =====\n")

    all_results = {}
    try:
        for label, chart_timeframe, tf_cal_override in config.BACKTEST_CONFIGS:
            print(f"===== Config: {label} =====")
            t0 = time.time()
            trades, durations_min, processed_count = await run_config_backtest(client, label, chart_timeframe, tf_cal_override)
            all_results[label] = (trades, durations_min)
            print(f"[{label}] Done: {len(trades)} scorable trade(s) in {time.time()-t0:.1f}s.")
            # Per-config completion email -- fires once, right when THIS
            # config's own loop finishes (not just the single combined
            # email at the very end of all 3 configs).
            state = ckpt.load_checkpoint(label)
            total_bars = len(state.get("flagged_bar_epochs", [])) if state else 0
            send_email(label, processed_count, total_bars, len(trades), event="done")
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

    print("\n\n===== REPAINT / TRAJECTORY ANALYSIS =====")
    repaint_result = repaint_analysis.generate_report()
    print(repaint_result["report_text"])
    if repaint_result["has_data"]:
        repaint_result["full_df"].to_csv("repaint_analysis_detail.csv", index=False)
        repaint_result["repaint_summary"].to_csv("repaint_analysis_summary.csv", index=False)
        repaint_result["flicker_summary"].to_csv("repaint_analysis_flicker_summary.csv", index=False)
        repaint_result["pre_summary"].to_csv("repaint_analysis_pre_buildup_summary.csv", index=False)
        repaint_result["post_summary"].to_csv("repaint_analysis_post_survival_summary.csv", index=False)
        print("\nSaved: repaint_analysis_detail.csv + 4 summary CSVs")

        send_raw_email("[Backtest] repaint / trajectory analysis", repaint_result["report_text"])

        repaint_csvs = sorted(glob.glob("repaint_analysis_*.csv"))
        if repaint_csvs:
            send_email_with_attachments(
                "[Backtest] repaint analysis attachments",
                "Attached are the repaint/trajectory analysis CSVs:\n\n"
                + "\n".join(f"- {p}" for p in repaint_csvs),
                repaint_csvs,
            )
    else:
        print("No repaint analysis data available (no trades with cluster trajectories yet).")

    print("\n\n===== COMBO RESULTS (mutually exclusive, sums to total trades) =====")
    print(combo_text)
    print("\n\n===== PER-CATEGORY RESULTS (inclusive, overlaps counted in each) =====")
    print(category_text)
    print("\nSaved: backtest_combo_results.csv, backtest_category_results.csv")


if __name__ == "__main__":
    asyncio.run(main())
