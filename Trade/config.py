"""
config.py
Central place for API credentials and default parameters.
DO NOT commit this file with a real token to any public repo.
"""

# ── Deriv API ──
# app_id 1089 is Deriv's public demo/test app id, fine for personal scripts.
# Get your own at https://developers.deriv.com/docs/, though 1089 works.
APP_ID = 1089
API_TOKEN = "mA3jGstqSBLZ357"  # demo account token — Read + Trade scope only
WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}"

# ── Market ──
SYMBOL = "R_100"          # Volatility 100 Index (standard, 2s ticks)

# ── Chart timeframe / HMM calibration ──────────────────────────────────────
# CHART_TIMEFRAME: "5m" or "15m" -- controls candle granularity pulled from
# Deriv and fed into every indicator.
CHART_TIMEFRAME = "5m"

# TF_CAL_OVERRIDE: leave as None to auto-match the HMM calibration to
# CHART_TIMEFRAME ("5m" -> "M5 Trained", "15m" -> "M15 Trained"). Set it
# explicitly to "M5 Trained" or "M15 Trained" to force a deliberate
# mismatch (e.g. testing "M15 Trained" while charting 5m).
TF_CAL_OVERRIDE = None

_TIMEFRAME_SECONDS = {"5m": 300, "15m": 900}
_AUTO_TF_CAL = {"5m": "M5 Trained", "15m": "M15 Trained"}


def resolve_granularity_and_tf_cal(chart_timeframe: str = None, tf_cal_override: str = None):
    """
    Returns (granularity_seconds, tf_cal) resolved from CHART_TIMEFRAME +
    TF_CAL_OVERRIDE (or explicit args, for running multiple configs in the
    same process without mutating module globals -- see backtest.py).
    """
    tf = chart_timeframe or CHART_TIMEFRAME
    if tf not in _TIMEFRAME_SECONDS:
        raise ValueError(f"CHART_TIMEFRAME must be '5m' or '15m', got {tf!r}")
    granularity = _TIMEFRAME_SECONDS[tf]

    override = tf_cal_override if tf_cal_override is not None else TF_CAL_OVERRIDE
    tf_cal = override if override is not None else _AUTO_TF_CAL[tf]
    if tf_cal not in ("M5 Trained", "M15 Trained"):
        raise ValueError(f"tf_cal must be 'M5 Trained' or 'M15 Trained', got {tf_cal!r}")

    return granularity, tf_cal


# ── MoVol HMM model ──
MOVOL_MODEL = "S=4 (Dir. x Vol)"         # "S=4 (Dir. x Vol)" | "S=5 (Neutral/Chop)" | "S=6 (Quiet/Volatile Chop)"
MOVOL_LOOKBACK_HOURS = 3.0

# ── Strategy ──
SIGNAL_THRESHOLD = 4

# ── Trading (placeholder — money management not yet built) ──
DEFAULT_STAKE = 1.0       # fixed stake in USD until money_management.py exists
CONTRACT_DURATION = 1     # single-contract default for live trading (not the backtest sweep)
CONTRACT_DURATION_UNIT = "m"

# ── History buffer (live warm-up) ──
CANDLE_HISTORY_COUNT = 1000   # how many bars to pull for indicator warm-up

# ── Backtest settings ───────────────────────────────────────────────────────
BACKTEST_HISTORY_WEEKS = 3/7   # 3 days, for a quick sanity-check run

# Set to a number (e.g. 1000) to pull an exact candle count instead of
# computing it from BACKTEST_HISTORY_WEEKS. Leave as None to use the
# weeks-based calculation. Useful for quick sanity-check runs.
BACKTEST_HISTORY_CANDLES = 1000

# Expiration durations to test, in minutes, per chart timeframe.
BACKTEST_DURATIONS_MIN = {
    "5m": [3, 4, 5, 6, 7, 8, 9, 10],
    "15m": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
}

# A tie at expiration (exit price == entry price) counts as a loss for
# both CALL and PUT, matching how Deriv actually settles Rise/Fall contracts.
TIE_COUNTS_AS_LOSS = True

# Three configs to run: (label, chart_timeframe, tf_cal_override)
# tf_cal_override=None means auto-match.
BACKTEST_CONFIGS = [
    ("5m_auto_M5",     "5m",  None),            # 5m chart, auto-matched M5 Trained
    ("15m_auto_M15",   "15m", None),            # 15m chart, auto-matched M15 Trained
    ("5m_mismatch_M15", "5m", "M15 Trained"),   # 5m chart, deliberately using M15 Trained
]

# Signals near the end of the pulled candle window that can't be fully
# resolved out to their longest tested expiration are dropped rather than
# scored on incomplete data.
DROP_EDGE_OF_DATA_SIGNALS = True
