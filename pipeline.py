"""
pipeline.py
Wires indicators.py + hmm_model.py + strategy.py together.
Takes a DataFrame of OHLCV bars (oldest -> newest) and a config dict,
returns a DataFrame with all intermediate signals plus a 'decision' column
('CALL' / 'PUT' / None) per bar.

Used by both backtest.py (batch) and live_trader.py (rolling buffer,
recomputed on each new completed candle).
"""

import pandas as pd
import numpy as np

import indicators as ind
import hmm_model as hmm
from strategy import SignalEngine, StrategyConfig, build_movol_gate_column


DEFAULT_MVT_CFG = dict(
    zi_window=3, zi_volume_weighted=True, zi_normalize=True,
    vpmo_len1=2, vpmo_len2=2, vpmo_threshold=0.0015,
    prt_pivot_len=2, prt_atr_len=3, prt_vol_len=3, prt_thresh=0.1,
    vel_ma_len=9, vel_norm_period=35, vel_ma_type="EMA",
    vel_threshold=1.0, acc_threshold=2.0,
    seconds_per_bar=60,
)

DEFAULT_EPSR_CFG = dict(
    pi_len=3, e_len=3, e_atr_len=3, phi_len=3,
    epsr_norm_period=10, smooth_len=3,
    upper_t=0.5, lower_t=-0.5, pi_breakout_t=0.25,
)


def run_pipeline(df: pd.DataFrame, tf_cal="M5 Trained", movol_model="S=4 (Dir. x Vol)",
                  movol_lookback_hours=3.0, seconds_per_bar=60,
                  mvt_cfg=None, epsr_cfg=None, strategy_cfg: StrategyConfig = None,
                  mintick=0.00001, cisd_min_improve_1=10, cisd_min_improve_2=140,
                  supertrend_mult=1.0, supertrend_period=9):
    """
    df must have columns: open, high, low, close, volume (oldest -> newest, reset index).
    Returns a copy of df with all signal/indicator columns + 'decision'.
    """
    mvt_cfg = {**DEFAULT_MVT_CFG, **(mvt_cfg or {})}
    mvt_cfg["seconds_per_bar"] = seconds_per_bar
    epsr_cfg = {**DEFAULT_EPSR_CFG, **(epsr_cfg or {})}
    strategy_cfg = strategy_cfg or StrategyConfig()

    df = df.reset_index(drop=True).copy()

    # ── MVT oscillators + discrete signal flags ──
    mvt_sig = ind.compute_mvt_signals(df, mvt_cfg)
    df = pd.concat([df, mvt_sig], axis=1)

    # ── HMM observables + MVT 5-state HMM forward filter ──
    hmm_obs = ind.compute_hmm_observables(df, seconds_per_bar)
    mvt_hmm = hmm.MVTHmm(tf_cal=tf_cal)
    mvt_steps = mvt_hmm.run(hmm_obs)
    df["mvt_dom_code"] = [s["dominant_code"] if s else None for s in mvt_steps]
    df["mvt_confidence"] = [s["confidence"] if s else None for s in mvt_steps]

    # ── MoVol HMM ──
    movol_length = max(5, round(movol_lookback_hours * 3600.0 / seconds_per_bar))
    movol_obs = hmm.compute_movol_observables(df, movol_length)
    movol_hmm = hmm.MoVolHmm(model_sel=movol_model, tf_cal=tf_cal)
    movol_steps = movol_hmm.run(movol_obs["obs_mom"].values, movol_obs["obs_vol"].values)
    df["movol_dom_dir"] = [s["dominant_dir"] if s else None for s in movol_steps]
    df["movol_dom_label"] = [s["dominant_label"] if s else None for s in movol_steps]

    # ── MoVol-gated bar colour direction (+1/-1/0) ──
    df["movol_gate_dir"] = build_movol_gate_column(df["mvt_dom_code"].tolist(), df["movol_dom_dir"].tolist())

    # ── EPSR / pi-breakout ──
    epsr = ind.compute_epsr(df, epsr_cfg)
    df = pd.concat([df, epsr], axis=1)

    # ── SuperTrend ──
    st = ind.compute_supertrend(df, mult=supertrend_mult, period=supertrend_period)
    df = pd.concat([df, st], axis=1)

    # ── CISD Dual Trailer ──
    cisd = ind.CISDDualTrailer(min_improve_1=cisd_min_improve_1, min_improve_2=cisd_min_improve_2,
                                mintick=mintick)
    cisd_out = cisd.run(df)
    df = pd.concat([df, cisd_out], axis=1)

    # ── previous-bar shifted columns needed by Category A signals + Category 3 ──
    for col in ["velbuy", "velsell", "acc_up", "acc_down",
                "cisd_bull_flip1", "cisd_bear_flip1", "cisd_bull_flip2", "cisd_bear_flip2"]:
        df[f"{col}_prev"] = df[col].shift(1).fillna(False)

    df["mvt_dom_code_prev1"] = df["mvt_dom_code"].shift(1)
    df["mvt_dom_code_prev2"] = df["mvt_dom_code"].shift(2)
    df["epsr_norm_prev"] = df["epsr_norm"].shift(1)

    # ── run strategy bar by bar ──
    engine = SignalEngine(strategy_cfg)
    decisions, buy_counts, sell_counts = [], [], []
    buy_hits_list, sell_hits_list, buy_cats_list, sell_cats_list = [], [], [], []
    for i, row in df.iterrows():
        result = engine.evaluate_bar(row, i)
        decisions.append(result["decision"])
        buy_counts.append(result["buy_count"])
        sell_counts.append(result["sell_count"])
        buy_hits_list.append(result["buy_hits"])
        sell_hits_list.append(result["sell_hits"])
        buy_cats_list.append(result["buy_categories"])
        sell_cats_list.append(result["sell_categories"])

    df["decision"] = decisions
    df["buy_count"] = buy_counts
    df["sell_count"] = sell_counts
    df["buy_hits"] = buy_hits_list
    df["sell_hits"] = sell_hits_list
    df["buy_categories"] = buy_cats_list
    df["sell_categories"] = sell_cats_list

    return df
