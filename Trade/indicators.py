"""
indicators.py
Ports of: Zonal Imbalance, VPMO, PRT, Velocity (MVT Signal Pro),
          EPSR / pi-breakout (MoSurge SR), SuperTrend, CISD Dual Trailer.

Design: all functions take a pandas DataFrame with columns
['open','high','low','close','volume'] and return either a Series
or add columns to a copy of the DataFrame. Recompute on a rolling
buffer of bars for live use (simple + correct, not the fastest).
"""

import numpy as np
import pandas as pd

PI = np.pi
E = np.e
PHI = 2.0 * np.cos(PI / 5.0)


# ─── generic rolling helpers (mirror Pine ta.* functions) ────────────────────

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length).mean()


def stdev(series: pd.Series, length: int) -> pd.Series:
    # Pine's ta.stdev uses population stdev (ddof=0)
    return series.rolling(length).std(ddof=0)


def rma(series: pd.Series, length: int) -> pd.Series:
    # Wilder's moving average, same as ta.rma in Pine
    return series.ewm(alpha=1.0 / length, adjust=False).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr


def atr(df: pd.DataFrame, length: int) -> pd.Series:
    return rma(true_range(df), length)


def zscore(series: pd.Series, period: int) -> pd.Series:
    avg = sma(series, period)
    std = stdev(series, period)
    std = std.replace(0, 1)
    return (series - avg) / std


def ma_dispatch(series: pd.Series, length: int, ma_type: str) -> pd.Series:
    if ma_type == "SMA":
        return sma(series, length)
    if ma_type == "EMA":
        return ema(series, length)
    if ma_type == "WMA":
        weights = np.arange(1, length + 1)
        return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    if ma_type == "RMA":
        return rma(series, length)
    return ema(series, length)


def timeframe_minutes(seconds_per_bar: float) -> float:
    return seconds_per_bar / 60.0


# ─── ZONAL IMBALANCE ──────────────────────────────────────────────────────────

def compute_zio(df, window=3, volume_weighted=True, normalize=True):
    delta = df["close"] - df["open"]
    if volume_weighted:
        bullish = np.where(delta > 0, delta * df["volume"], 0.0)
        bearish = np.where(delta < 0, -delta * df["volume"], 0.0)
    else:
        bullish = np.where(delta > 0, delta, 0.0)
        bearish = np.where(delta < 0, -delta, 0.0)
    bull_sum = sma(pd.Series(bullish, index=df.index), window)
    bear_sum = sma(pd.Series(bearish, index=df.index), window)
    raw_imbalance = bull_sum - bear_sum
    total = bull_sum + bear_sum + 1e-6
    if normalize:
        return (raw_imbalance / total) * 100
    return raw_imbalance


# ─── VPMO ─────────────────────────────────────────────────────────────────────

def compute_vpmo(df, length1=2, length2=2):
    price_move = df["close"].diff()
    vol_weight = price_move * df["volume"]
    smoothed1 = ema(vol_weight, length1)
    vpmo_raw = ema(smoothed1, length2)
    abs_vol_weight = price_move.abs() * df["volume"]
    abs_smoothed1 = ema(abs_vol_weight, length1)
    avg_abs = ema(abs_smoothed1, length2) + 1e-6
    return vpmo_raw / avg_abs


# ─── PRT (Pivot-Range Thrust) ──────────────────────────────────────────────────

def compute_prt(df, pivot_len=2, atr_len=3, vol_len=3):
    pivot_high = df["high"].shift(1).rolling(pivot_len).max()
    pivot_low = df["low"].shift(1).rolling(pivot_len).min()
    dist_bull = (df["close"] - pivot_high).clip(lower=0)
    dist_bear = (pivot_low - df["close"]).clip(lower=0)
    atr_val = atr(df, atr_len).replace(0, np.nan)
    norm_bull = (dist_bull / atr_val).fillna(0)
    norm_bear = (dist_bear / atr_val).fillna(0)
    vol_ratio = df["volume"] / sma(df["volume"], vol_len)
    return (norm_bull - norm_bear) * vol_ratio


# ─── VELOCITY ─────────────────────────────────────────────────────────────────

def compute_velocity(df, ma_length=9, norm_period=35, ma_type="EMA", seconds_per_bar=60):
    time_multiplier = timeframe_minutes(seconds_per_bar)
    price_change = df["close"].diff()
    raw_velocity = price_change / time_multiplier
    smooth_velocity = ma_dispatch(raw_velocity, ma_length, ma_type)
    return zscore(smooth_velocity, norm_period)


# ─── HMM OBSERVABLES (fixed params — exact replication of MVT defaults) ──────

def compute_hmm_observables(df, seconds_per_bar=60):
    delta = df["close"] - df["open"]
    bull = np.where(delta > 0, delta * df["volume"], 0.0)
    bear = np.where(delta < 0, -delta * df["volume"], 0.0)
    b_sum = sma(pd.Series(bull, index=df.index), 3)
    e_sum = sma(pd.Series(bear, index=df.index), 3)
    total = b_sum + e_sum + 1e-6
    zio_hmm = ((b_sum - e_sum) / total) * 100.0

    pm = df["close"].diff()
    vw = pm * df["volume"]
    s1 = ema(vw, 2)
    vpmo_r = ema(s1, 2)
    avw = pm.abs() * df["volume"]
    as1 = ema(avw, 2)
    aavg = ema(as1, 2) + 1e-6
    vpmo_hmm = vpmo_r / aavg

    ph = df["high"].shift(1).rolling(2).max()
    pl = df["low"].shift(1).rolling(2).min()
    d_bull = (df["close"] - ph).clip(lower=0)
    d_bear = (pl - df["close"]).clip(lower=0)
    patr = atr(df, 3)
    patr_s = patr.where(patr != 0, 1.0)
    pvr = df["volume"] / (sma(df["volume"], 3) + 1e-6)
    prt_hmm = (d_bull / patr_s - d_bear / patr_s) * pvr

    rv = pm / (seconds_per_bar / 60.0)
    sv = ema(rv, 9)
    sv_std = stdev(sv, 35)
    sv_mn = sma(sv, 35)
    vel_hmm = ((sv - sv_mn) / sv_std.replace(0, np.nan)).fillna(0.0)

    return pd.DataFrame({
        "zio_hmm": zio_hmm, "vpmo_hmm": vpmo_hmm,
        "prt_hmm": prt_hmm, "vel_hmm": vel_hmm,
    })


# ─── MVT SIGNAL FLAGS (bull/bear condition booleans) ──────────────────────────

def compute_mvt_signals(df, cfg):
    zio_osc = compute_zio(df, cfg["zi_window"], cfg["zi_volume_weighted"], cfg["zi_normalize"])
    vpmo_osc = compute_vpmo(df, cfg["vpmo_len1"], cfg["vpmo_len2"])
    prt_osc = compute_prt(df, cfg["prt_pivot_len"], cfg["prt_atr_len"], cfg["prt_vol_len"])
    vel_osc = compute_velocity(df, cfg["vel_ma_len"], cfg["vel_norm_period"],
                                cfg["vel_ma_type"], cfg["seconds_per_bar"])

    zonalup = zio_osc > 0
    zonaldown = zio_osc < 0
    buyvpmo = vpmo_osc > cfg["vpmo_threshold"]
    sellvpmo = vpmo_osc < -cfg["vpmo_threshold"]

    prt_long_sig = prt_osc > cfg["prt_thresh"]
    prt_short_sig = prt_osc < -cfg["prt_thresh"]
    prt_long = prt_long_sig & ~prt_long_sig.shift(1).fillna(False)
    prt_short = prt_short_sig & ~prt_short_sig.shift(1).fillna(False)
    prt_prevbar = prt_osc.shift(1) == 0

    upvel = vel_osc > vel_osc.shift(1)
    downvel = vel_osc < vel_osc.shift(1)
    upvelsig = upvel & (vel_osc > cfg["vel_threshold"])
    downvelsig = downvel & (vel_osc < -cfg["vel_threshold"])

    full_buy = zonalup & buyvpmo & prt_long
    full_sell = zonaldown & sellvpmo & prt_short
    final_full_buy = full_buy & ~full_buy.shift(1).fillna(False)
    final_full_sell = full_sell & ~full_sell.shift(1).fillna(False)

    longrvs = (prt_short_sig.shift(1).fillna(False) & prt_long_sig) | \
              (prt_short_sig.shift(2).fillna(False) & prt_prevbar & prt_long_sig)
    shortrvs = (prt_long_sig.shift(1).fillna(False) & prt_short_sig) | \
               (prt_long_sig.shift(2).fillna(False) & prt_prevbar & prt_short_sig)

    velbuy = upvelsig & (prt_long_sig & (prt_osc > prt_osc.shift(1) / 2)) & \
             (buyvpmo & (vpmo_osc > vpmo_osc.shift(1)))
    velsell = downvelsig & (prt_short_sig & (prt_osc < prt_osc.shift(1) / 2)) & \
              (sellvpmo & (vpmo_osc < vpmo_osc.shift(1)))

    acc_up = (vel_osc.shift(1) < cfg["vel_threshold"]) & (vel_osc >= cfg["vel_threshold"]) & \
             (vel_osc.shift(1) < cfg["acc_threshold"]) & (vel_osc >= cfg["acc_threshold"])
    acc_down = (vel_osc.shift(1) > -cfg["vel_threshold"]) & (vel_osc <= -cfg["vel_threshold"]) & \
               (vel_osc.shift(1) > -cfg["acc_threshold"]) & (vel_osc <= -cfg["acc_threshold"])

    return pd.DataFrame({
        "zio_osc": zio_osc, "vpmo_osc": vpmo_osc, "prt_osc": prt_osc, "vel_osc": vel_osc,
        "final_full_buy": final_full_buy.fillna(False),
        "final_full_sell": final_full_sell.fillna(False),
        "longrvs": longrvs.fillna(False),
        "shortrvs": shortrvs.fillna(False),
        "velbuy": velbuy.fillna(False),
        "velsell": velsell.fillna(False),
        "acc_up": acc_up.fillna(False),
        "acc_down": acc_down.fillna(False),
    })


# ─── EPSR (Euler-Phi Structural Resonance) ────────────────────────────────────

def compute_epsr(df, cfg):
    pi_high = df["high"].shift(1).rolling(cfg["pi_len"]).max()
    pi_low = df["low"].shift(1).rolling(cfg["pi_len"]).min()
    pi_range = pi_high - pi_low
    bull_break = (df["close"] - pi_high).clip(lower=0)
    bear_break = (pi_low - df["close"]).clip(lower=0)
    net_break = bull_break - bear_break
    pi_norm = (net_break / pi_range.replace(0, np.nan)).fillna(0.0)
    pi_norm = pi_norm.clip(-1.0, 1.0)
    pi_breakout = np.sin(pi_norm * PI / 2.0)

    atr_val = atr(df, cfg["e_atr_len"])
    raw_mom = df["close"] - df["close"].shift(cfg["e_len"])
    norm_mom = (raw_mom / atr_val.replace(0, np.nan)).fillna(0.0)
    e_thrust = np.sign(norm_mom) * np.minimum(np.power(E, norm_mom.abs()) - 1.0, 3.0)

    phi_high = df["high"].shift(1).rolling(cfg["phi_len"]).max()
    phi_low = df["low"].shift(1).rolling(cfg["phi_len"]).min()
    bull_pen = (df["close"] - phi_high).clip(lower=0)
    bear_pen = (phi_low - df["close"]).clip(lower=0)
    bar_range = (df["high"] - df["low"]).replace(0, np.nan)
    bull_pen_pct = (bull_pen / bar_range).fillna(0.0)
    bear_pen_pct = (bear_pen / bar_range).fillna(0.0)
    phi_penetration = (bull_pen_pct - bear_pen_pct) * PHI

    epsr_raw = e_thrust * pi_breakout * phi_penetration
    epsr_mean = sma(epsr_raw, cfg["epsr_norm_period"])
    epsr_std = stdev(epsr_raw, cfg["epsr_norm_period"]).replace(0, np.nan)
    epsr_norm = ((epsr_raw - epsr_mean) / epsr_std).fillna(0.0)

    pi_breakout_bull_sig = pi_breakout > cfg["pi_breakout_t"]
    pi_breakout_bear_sig = pi_breakout < -cfg["pi_breakout_t"]

    epsr_bull = (epsr_norm.shift(1) < epsr_norm) & (epsr_norm > cfg["upper_t"])
    epsr_bear = (epsr_norm.shift(1) > epsr_norm) & (epsr_norm < cfg["lower_t"])

    bull_thrust = epsr_bull & pi_breakout_bull_sig
    bear_thrust = epsr_bear & pi_breakout_bear_sig

    return pd.DataFrame({
        "pi_breakout": pi_breakout,
        "epsr_norm": epsr_norm,
        "bull_thrust": bull_thrust.fillna(False),
        "bear_thrust": bear_thrust.fillna(False),
    })


# ─── SUPERTREND ────────────────────────────────────────────────────────────────

def compute_supertrend(df, mult=1.0, period=9):
    atr_val = atr(df, period)
    hl2 = (df["high"] + df["low"]) / 2.0
    upperband = hl2 + mult * atr_val
    lowerband = hl2 - mult * atr_val

    n = len(df)
    final_upper = np.zeros(n)
    final_lower = np.zeros(n)
    supertrend = np.zeros(n)
    direction = np.ones(n, dtype=int)  # -1 bullish, 1 bearish (matches Pine convention)

    close = df["close"].values
    ub = upperband.values
    lb = lowerband.values

    final_upper[0] = ub[0]
    final_lower[0] = lb[0]
    supertrend[0] = ub[0]
    direction[0] = 1

    for i in range(1, n):
        final_upper[i] = ub[i] if (ub[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]) else final_upper[i-1]
        final_lower[i] = lb[i] if (lb[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]) else final_lower[i-1]

        if supertrend[i-1] == final_upper[i-1]:
            direction[i] = -1 if close[i] > final_upper[i] else 1
        else:
            direction[i] = -1 if close[i] < final_lower[i] else 1
        # NOTE: mirrors ta.supertrend flip semantics closely enough for signal use
        supertrend[i] = final_lower[i] if direction[i] == -1 else final_upper[i]

    st = pd.Series(supertrend, index=df.index)
    dirn = pd.Series(direction, index=df.index)
    signal_buy = (df["close"] > st) & (df["close"].shift(1) <= st.shift(1))
    signal_sell = (df["close"] < st) & (df["close"].shift(1) >= st.shift(1))

    return pd.DataFrame({
        "supertrend": st, "st_dir": dirn,
        "st_signal_buy": signal_buy.fillna(False),
        "st_signal_sell": signal_sell.fillna(False),
    })


# ─── CISD DUAL TRAILER (stateful, bar-by-bar port) ────────────────────────────

class CISDDualTrailer:
    """
    Faithful bar-by-bar port of the CISD Dual Trailer state machine.
    Call `.update(df)` with the full available OHLC history (open/close arrays);
    it processes sequentially and returns per-bar state + flip flags.
    """

    def __init__(self, min_improve_1=10, min_improve_2=140, mintick=0.001, max_lookback=500):
        self.min_improve_1 = min_improve_1
        self.min_improve_2 = min_improve_2
        self.mintick = mintick
        self.max_lookback = max_lookback

    @staticmethod
    def _attitude(o, c):
        if c > o:
            return 1
        if c < o:
            return -1
        return 0

    def _find_run_origin(self, opens, closes, idx, bias):
        """Scan backward from idx while candles match bias. Returns (level, origin_idx)."""
        first_att = self._attitude(opens[idx], closes[idx])
        extreme_open = opens[idx]
        extreme_idx = idx
        if first_att != 0 and first_att == bias:
            run_len = 0
            lo = max(0, idx - self.max_lookback)
            for i in range(idx - 1, lo - 1, -1):
                att = self._attitude(opens[i], closes[i])
                if att == 0:
                    continue
                if att != bias:
                    break
                run_len = idx - i
                if bias == 1:
                    if opens[i] < extreme_open:
                        extreme_open = opens[i]
                        extreme_idx = i
                else:
                    if opens[i] > extreme_open:
                        extreme_open = opens[i]
                        extreme_idx = i
        return extreme_open, extreme_idx

    def _find_prior_run_origin(self, opens, closes, idx, bias):
        found = False
        run_end = None
        extreme_open = None
        lo = max(0, idx - self.max_lookback)
        for i in range(idx - 1, lo - 1, -1):
            att = self._attitude(opens[i], closes[i])
            if att == 0:
                continue
            matches = att == bias
            if not found:
                if matches:
                    found = True
                    run_end = i
                    extreme_open = opens[i]
            else:
                if not matches:
                    break
                run_end = i
                if bias == 1:
                    extreme_open = min(extreme_open, opens[i])
                else:
                    extreme_open = max(extreme_open, opens[i])
        if run_end is None:
            return None, None
        extreme_idx = run_end
        for k in range(run_end, idx):
            if opens[k] == extreme_open:
                extreme_idx = k
                break
        return extreme_open, extreme_idx

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        opens = df["open"].values
        closes = df["close"].values
        highs = df["high"].values
        lows = df["low"].values
        n = len(df)

        state1 = np.zeros(n, dtype=int)
        state2 = np.zeros(n, dtype=int)
        origin1 = np.full(n, np.nan)
        origin2 = np.full(n, np.nan)
        bull_flip1 = np.zeros(n, dtype=bool)
        bear_flip1 = np.zeros(n, dtype=bool)
        bull_flip2 = np.zeros(n, dtype=bool)
        bear_flip2 = np.zeros(n, dtype=bool)

        ts1 = ts2 = 0
        ol1 = ol2 = np.nan
        wm1 = wm2 = np.nan
        min_imp1 = self.min_improve_1 * self.mintick
        min_imp2 = self.min_improve_2 * self.mintick

        for i in range(n):
            if ts1 == 0 and i > 10:
                att0 = self._attitude(opens[i], closes[i])
                if att0 == 0:
                    for k in range(i - 1, max(-1, i - 51), -1):
                        att0 = self._attitude(opens[k], closes[k])
                        if att0 != 0:
                            break
                if att0 != 0:
                    lvl0, _ = self._find_run_origin(opens, closes, i, att0)
                    ts1 = ts2 = att0
                    ol1 = ol2 = lvl0
                    wm1 = wm2 = (highs[i] if att0 == 1 else lows[i])

            att = self._attitude(opens[i], closes[i])

            # trailer 1 re-anchor
            if ts1 == 1:
                if highs[i] > wm1:
                    wm1 = highs[i]
                    lvl, _ = (self._find_run_origin(opens, closes, i, 1) if att == 1
                              else self._find_prior_run_origin(opens, closes, i, 1))
                    if lvl is not None:
                        ol1 = lvl
            elif ts1 == -1:
                if lows[i] < wm1:
                    wm1 = lows[i]
                    lvl, _ = (self._find_run_origin(opens, closes, i, -1) if att == -1
                              else self._find_prior_run_origin(opens, closes, i, -1))
                    if lvl is not None:
                        ol1 = lvl

            if ts1 != 0:
                if ts1 == 1:
                    lvl, _ = self._find_prior_run_origin(opens, closes, i, 1)
                    if lvl is not None and (lvl - ol1) >= min_imp1:
                        ol1 = lvl
                else:
                    lvl, _ = self._find_prior_run_origin(opens, closes, i, -1)
                    if lvl is not None and (ol1 - lvl) >= min_imp1:
                        ol1 = lvl

            bf1 = ts1 == -1 and closes[i] > ol1
            brf1 = ts1 == 1 and closes[i] < ol1
            if bf1 or brf1:
                new_state = 1 if bf1 else -1
                ts1 = new_state
                lvl2, _ = self._find_run_origin(opens, closes, i, new_state)
                ol1 = lvl2
                wm1 = highs[i] if new_state == 1 else lows[i]
            bull_flip1[i] = bf1
            bear_flip1[i] = brf1

            # trailer 2 re-anchor
            if ts2 == 1:
                if highs[i] > wm2:
                    wm2 = highs[i]
                    lvl, _ = (self._find_run_origin(opens, closes, i, 1) if att == 1
                              else self._find_prior_run_origin(opens, closes, i, 1))
                    if lvl is not None:
                        ol2 = lvl
            elif ts2 == -1:
                if lows[i] < wm2:
                    wm2 = lows[i]
                    lvl, _ = (self._find_run_origin(opens, closes, i, -1) if att == -1
                              else self._find_prior_run_origin(opens, closes, i, -1))
                    if lvl is not None:
                        ol2 = lvl

            if ts2 != 0:
                if ts2 == 1:
                    lvl, _ = self._find_prior_run_origin(opens, closes, i, 1)
                    if lvl is not None and (lvl - ol2) >= min_imp2:
                        ol2 = lvl
                else:
                    lvl, _ = self._find_prior_run_origin(opens, closes, i, -1)
                    if lvl is not None and (ol2 - lvl) >= min_imp2:
                        ol2 = lvl

            bf2 = ts2 == -1 and closes[i] > ol2
            brf2 = ts2 == 1 and closes[i] < ol2
            if bf2 or brf2:
                new_state = 1 if bf2 else -1
                ts2 = new_state
                lvl2b, _ = self._find_run_origin(opens, closes, i, new_state)
                ol2 = lvl2b
                wm2 = highs[i] if new_state == 1 else lows[i]
            bull_flip2[i] = bf2
            bear_flip2[i] = brf2

            state1[i] = ts1
            state2[i] = ts2
            origin1[i] = ol1
            origin2[i] = ol2

        return pd.DataFrame({
            "cisd_state1": state1, "cisd_state2": state2,
            "cisd_origin1": origin1, "cisd_origin2": origin2,
            "cisd_bull_flip1": bull_flip1, "cisd_bear_flip1": bear_flip1,
            "cisd_bull_flip2": bull_flip2, "cisd_bear_flip2": bear_flip2,
        }, index=df.index)
