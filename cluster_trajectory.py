"""
cluster_trajectory.py
Captures raw pre/cluster/post-window signal-count trajectories for
repaint analysis (see repaint_analysis.py for the interpretation layer
that reads this data).

Two ways to use this module:
  1. Imported by backtest.py -- capture_cluster_trajectories() runs
     automatically as part of every normal backtest run.
  2. Run directly -- `python cluster_trajectory.py` -- to backfill
     trajectories for EXISTING checkpoints (e.g. results produced before
     this feature existed, or clusters that were skipped on a prior
     partial resume), without re-running the rest of the backtest.
     Emails a summary when done, same reporting style as
     analyze_checkpoints.py / backtest.py's final email.
"""

import asyncio

import pandas as pd

import config
from deriv_client import DerivClient
from pipeline import run_pipeline
from strategy import StrategyConfig
from tick_replay import cluster_flagged_bars, compute_cluster_trajectory
from network_retry import call_with_reconnect, BacktestConnectionError, TICK_PACE_DELAY
import checkpoint as ckpt
from email_notifier import load_smtp_env_from_app_password, send_raw_email, is_configured


async def capture_cluster_trajectories(client: DerivClient, label: str, out: pd.DataFrame, df: pd.DataFrame,
                                        flagged_bar_epochs: list, existing_trajectory_ids: set, epoch_to_pos: dict,
                                        granularity: int, max_duration_sec: int, replay_pipeline_kwargs: dict,
                                        checkpoint_state: dict | None = None,
                                        processed_epoch_set: set | None = None,
                                        trades: list | None = None,
                                        skipped_bar_epochs: set | None = None,
                                        dropped_bar_epochs: set | None = None):
    """
    Additive pre-processing step. Groups flagged bars into same-direction
    clusters, and for every cluster whose trajectory hasn't already been
    captured (regardless of whether some/all of its bars were already
    scored in a prior run -- capture is idempotent per cluster, not
    gated by processing status), pulls ONE tick window spanning
    [pre-bar open, post-bar close + settlement buffer], and tracks the
    signal count continuously across pre-window -> cluster -> post-window.

    Does NOT determine entry timing or score win/loss -- that's still
    find_entry_tick/find_settlement_price in backtest.py, unchanged.
    This purely captures raw trajectory data for repaint_analysis.py.

    Saves a checkpoint after EVERY successfully captured cluster so that
    an interrupted run (connection death, process kill) preserves all
    clusters captured so far -- without this, a crash late in the loop
    loses every cluster captured in the current run. The checkpoint
    also carries the current Pass 2 state (processed/trades/skipped/
    dropped) so a resumed run picks up cleanly.

    Clusters sitting at the very edge of available history (no
    room for a pre/post bar) are skipped for now but tracked in
    `skipped_cluster_ids` so they can be retried on resume if the
    dataframe has grown (e.g. more candles pulled).

    Returns (cluster_trajectories: dict, epoch_to_cluster_id: dict).
    epoch_to_cluster_id covers EVERY cluster (newly captured or already
    existing), so any trade -- past or future -- can be linked to its
    cluster.
    """
    epoch_side_pairs = []
    for e in flagged_bar_epochs:
        pos = epoch_to_pos.get(e)
        if pos is None:
            continue
        row = out.iloc[pos]
        epoch_side_pairs.append((e, "buy" if row["decision"] == "CALL" else "sell"))

    clusters = cluster_flagged_bars(epoch_side_pairs, granularity)
    epoch_to_cluster_id = {e: str(c["epochs"][0]) for c in clusters for e in c["epochs"]}

    # Load previously skipped cluster IDs from checkpoint (if any)
    skipped_cluster_ids = set()
    if checkpoint_state is not None:
        skipped_cluster_ids = set(checkpoint_state.get("skipped_cluster_ids", []))

    # Build the list of clusters still needing capture, keeping each
    # cluster's ORIGINAL 1-based position in the full `clusters` list so
    # that progress prints (e.g. "cluster 18/21") reflect the cluster's
    # true position in the full set, not a sequential counter over the
    # remaining subset. Already-captured clusters (middle of the list)
    # are dropped; warmup_too_short clusters at the start and remaining
    # clusters at the end keep their original indices.
    clusters_needing_capture = [(orig_idx, c) for orig_idx, c in enumerate(clusters, start=1)
                                if str(c["epochs"][0]) not in existing_trajectory_ids]
    total_clusters = len(clusters)
    already_captured = total_clusters - len(clusters_needing_capture)
    if not clusters_needing_capture:
        if already_captured > 0:
            print(f"[{label}] All {total_clusters} cluster(s) already captured -- nothing to do.")
        return {}, epoch_to_cluster_id, skipped_cluster_ids

    total_bars = sum(len(c['epochs']) for _, c in clusters_needing_capture)
    buy_clusters = sum(1 for _, c in clusters_needing_capture if c["side"] == "buy")
    sell_clusters = sum(1 for _, c in clusters_needing_capture if c["side"] == "sell")
    if already_captured > 0:
        prior_bars = sum(len(c['epochs']) for c in clusters if str(c["epochs"][0]) in existing_trajectory_ids)
        prior_buy = sum(1 for c in clusters if c["side"] == "buy" and str(c["epochs"][0]) in existing_trajectory_ids)
        prior_sell = sum(1 for c in clusters if c["side"] == "sell" and str(c["epochs"][0]) in existing_trajectory_ids)
        print(f"[{label}] Captured trajectories of {already_captured} cluster(s) "
              f"from {prior_bars} Flagged bar(s) [Total: {prior_buy} buy, {prior_sell} sell]")
        print(f"[{label}] Remaining {len(clusters_needing_capture)} cluster(s) "
              f"from {total_bars} Flagged bar(s) [Total: {buy_clusters} buy, {sell_clusters} sell]")
        print(f"[{label}] Resuming...")
    else:
        print(f"[{label}] Capturing trajectories for {len(clusters_needing_capture)} cluster(s) "
              f"from {total_bars} Flagged bar(s) [Total: {buy_clusters} buy, {sell_clusters} sell]")

    cluster_trajectories = {}
    settlement_buffer = max_duration_sec + 30

    # Merge in trajectories from any prior (interrupted) run so the
    # per-cluster checkpoint saves carry forward everything captured so
    # far, not just the clusters captured in this invocation.
    if checkpoint_state is not None:
        prior_trajs = checkpoint_state.get("cluster_trajectories", {})
        cluster_trajectories = {str(k): v for k, v in prior_trajs.items()}

    # `i` is the cluster's original 1-based position in the full
    # `clusters` list (not a sequential counter over the remaining subset).
    for i, cluster in clusters_needing_capture:
        first_pos = epoch_to_pos[cluster["epochs"][0]]
        num_bars = len(cluster["epochs"])
        pre_pos = first_pos - 1
        post_pos = first_pos + num_bars

        cluster_id = str(cluster["epochs"][0])

        if pre_pos < 0 or post_pos >= len(df):
            # Edge of available history - no room for pre/post window.
            # Track this cluster as skipped so it can be retried on resume
            # if the dataframe has grown (e.g. more candles pulled).
            print(f"[{label}]   cluster {i}/{total_clusters} ({num_bars} bar(s), side={cluster['side']}) "
                  f"at edge of history (pre_pos={pre_pos}, post_pos={post_pos}, df_len={len(df)}) -- skipping for now.")
            skipped_cluster_ids.add(cluster_id)
            continue

        pre_bar_open = int(df.iloc[pre_pos]["epoch"])
        post_bar_close = int(df.iloc[post_pos]["epoch"]) + granularity
        try:
            ticks = await call_with_reconnect(
                client, label, f"cluster trajectory pull @{pre_bar_open}",
                client.get_tick_history, count=4000, start=pre_bar_open,
                end=post_bar_close + settlement_buffer)
        except BacktestConnectionError:
            print("connection could not be restored -- stopping trajectory capture, "
                  "already-processed trades are unaffected.")
            break
        except Exception as e:
            print(f"tick pull failed ({e}), skipping this cluster's trajectory.")
            continue
        await asyncio.sleep(TICK_PACE_DELAY)

        tick_tuples = [(t["epoch"], t["price"]) for t in ticks]
        traj = await compute_cluster_trajectory(out, first_pos, num_bars, tick_tuples, cluster["side"],
                                           granularity, replay_pipeline_kwargs)
        if traj is None:
            # compute_cluster_trajectory returned None -- only happens if ticks
            # is empty (Deriv has no tick data for this old time window).
            print(f"[{label}]   cluster {i}/{total_clusters} SKIPPED: no tick data from Deriv "
                  f"(retention limit -- ticks aged out).")
            continue
        if isinstance(traj, tuple):
            # Structured skip reason from compute_cluster_trajectory.
            reason_code, reason_msg = traj
            print(f"[{label}]   cluster {i}/{total_clusters} SKIPPED [{reason_code}]: {reason_msg}")
            continue

        print(f"[{label}]   cluster {i}/{total_clusters} ({len(traj['epochs'])} ticks).")
        # Store raw ticks for Pass 2 reuse (covers [pre_bar_open, post_bar_close + settlement_buffer])
        traj["ticks"] = ticks
        traj["window_start"] = pre_bar_open
        traj["window_end"] = post_bar_close + settlement_buffer
        cluster_trajectories[cluster_id] = traj

        # If this cluster was previously skipped (edge of history), remove it
        # from the skipped set since we've now captured it.
        skipped_cluster_ids.discard(cluster_id)

        # Save a checkpoint after every successfully captured cluster,
        # unconditionally -- including on a fresh run with no prior
        # checkpoint. Cluster counts are small and each pull is heavy
        # (4000-tick request + full per-tick pipeline recompute), so the
        # atomic save cost is negligible relative to the work it protects.
        # This also carries forward the current Pass 2 state so a
        # resumed run picks up cleanly if the process dies here.
        ckpt.save_checkpoint(
            label, flagged_bar_epochs, list(processed_epoch_set or []),
            trades or [],
            skipped_bar_epochs=list(skipped_bar_epochs or set()),
            dropped_bar_epochs=list(dropped_bar_epochs or set()),
            cluster_trajectories=cluster_trajectories,
            skipped_cluster_ids=list(skipped_cluster_ids))

    captured_count = len(cluster_trajectories) - len(existing_trajectory_ids)
    print(f"[{label}] Capturing Cluster Trajectories: Completed "
          f"({captured_count} captured this run, {already_captured} from prior run, "
          f"{len(skipped_cluster_ids)} skipped)")

    return cluster_trajectories, epoch_to_cluster_id, skipped_cluster_ids


async def backfill_config(client: DerivClient, label: str, chart_timeframe: str, tf_cal_override):
    """
    Standalone-mode backfill for one config: loads its existing candle
    cache + checkpoint (must already exist -- run a normal backtest
    first), reruns Pass 1 fresh, captures any not-yet-captured cluster
    trajectories, retroactively links existing trades, and saves.
    Returns a summary dict for the email report, or None if skipped.
    """
    df = ckpt.load_candles_cache(label)
    state = ckpt.load_checkpoint(label)
    if df is None or state is None:
        print(f"[{label}] No existing candle cache/checkpoint -- run a normal backtest first. Skipping.")
        return None

    df = df.sort_values("epoch").reset_index(drop=True)
    granularity, tf_cal = config.resolve_granularity_and_tf_cal(chart_timeframe, tf_cal_override)
    durations_min = config.BACKTEST_DURATIONS_MIN[chart_timeframe]
    max_duration_sec = max(durations_min) * 60

    pipeline_kwargs = dict(
        tf_cal=tf_cal, movol_model=config.MOVOL_MODEL, seconds_per_bar=granularity,
        movol_lookback_hours=config.MOVOL_LOOKBACK_HOURS,
        strategy_cfg=StrategyConfig(threshold=config.SIGNAL_THRESHOLD),
    )
    out = run_pipeline(df.drop(columns=["epoch"]), **pipeline_kwargs)
    out["epoch"] = df["epoch"].values
    epoch_to_pos = {int(e): i for i, e in enumerate(df["epoch"].values)}

    flagged_bar_epochs = state["flagged_bar_epochs"]
    existing_trajectory_ids = set(state.get("cluster_trajectories", {}).keys())
    trades = state["trades"]
    for t in trades:
        t["outcomes"] = {int(k): v for k, v in t["outcomes"].items()}

    before_count = len(existing_trajectory_ids)

    try:
        new_trajectories, epoch_to_cluster_id, skipped_cluster_ids = await capture_cluster_trajectories(
            client, label, out, df, flagged_bar_epochs, existing_trajectory_ids, epoch_to_pos,
            granularity, max_duration_sec, pipeline_kwargs,
            checkpoint_state=state,
            processed_epoch_set=set(state.get("processed_bar_epochs", [])),
            trades=trades,
            skipped_bar_epochs=set(state.get("skipped_bar_epochs", [])),
            dropped_bar_epochs=set(state.get("dropped_bar_epochs", [])))
    except BacktestConnectionError as e:
        print(f"[{label}] {e} -- partial progress (if any) has been saved.")
        new_trajectories, epoch_to_cluster_id, skipped_cluster_ids = {}, {}, set()

    cluster_trajectories = {**state.get("cluster_trajectories", {}), **new_trajectories}

    linked = 0
    for t in trades:
        be = int(t["bar_epoch"])
        if be in epoch_to_cluster_id and t.get("cluster_id") != epoch_to_cluster_id[be]:
            t["cluster_id"] = epoch_to_cluster_id[be]
            linked += 1

    ckpt.save_checkpoint(label, state["flagged_bar_epochs"], state["processed_bar_epochs"], trades,
                         skipped_bar_epochs=state.get("skipped_bar_epochs", []),
                         dropped_bar_epochs=state.get("dropped_bar_epochs", []),
                         cluster_trajectories=cluster_trajectories,
                         skipped_cluster_ids=list(skipped_cluster_ids))

    after_count = len(cluster_trajectories)
    total_trades_linked = sum(1 for t in trades if t.get("cluster_id") in cluster_trajectories)
    print(f"[{label}] Backfill complete: {after_count - before_count} new trajector(y/ies) captured "
          f"({after_count} total), {linked} trade(s) newly linked ({total_trades_linked} total linked).\n")

    return {
        "label": label, "new_trajectories": after_count - before_count,
        "total_trajectories": after_count, "newly_linked_trades": linked,
        "total_linked_trades": total_trades_linked, "total_trades": len(trades),
    }


async def main():
    load_smtp_env_from_app_password()

    client = DerivClient()
    await client.connect()
    auth = await client.authorize()
    print(f"Authorized as {auth.get('loginid')} (is_virtual={auth.get('is_virtual')})\n")

    summaries = []
    try:
        for label, chart_timeframe, tf_cal_override in config.BACKTEST_CONFIGS:
            print(f"===== Config: {label} =====")
            result = await backfill_config(client, label, chart_timeframe, tf_cal_override)
            if result:
                summaries.append(result)
    except KeyboardInterrupt:
        print("\n\nInterrupted -- any cluster fully captured before this point has already been saved. "
              "Just re-run `python cluster_trajectory.py` to continue backfilling the rest.")
    finally:
        await client.close()

    if not summaries:
        print("No configs had existing checkpoints to backfill.")
        return

    report_lines = ["Cluster trajectory backfill summary.", ""]
    for s in summaries:
        report_lines.append(f"{s['label']}:")
        report_lines.append(f"  New trajectories captured this run: {s['new_trajectories']}")
        report_lines.append(f"  Total trajectories stored: {s['total_trajectories']}")
        report_lines.append(f"  Trades newly linked this run: {s['newly_linked_trades']}")
        report_lines.append(f"  Total trades linked / total trades: "
                            f"{s['total_linked_trades']}/{s['total_trades']}")
        report_lines.append("")

    body = "\n".join(report_lines)
    print("\n" + body)

    if is_configured():
        ok = send_raw_email("[Backtest] cluster trajectory backfill summary", body)
        print(f"\nEmail sent: {ok}")
    else:
        print("\nEmail skipped: SMTP not configured.")


if __name__ == "__main__":
    asyncio.run(main())
