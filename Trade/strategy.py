"""
strategy.py
Combines MVT signals, EPSR thrust, SuperTrend, MoVol-gated bar colour, and
CISD Dual Trailer flips into a single Call/Put decision per bar (works for
both fully-closed historical bars in backtest and the live/forming bar in
real-time trading -- see evaluate_bar's `bar_index` semantics below).

THREE INDEPENDENT SIGNAL CATEGORIES. Any one of them qualifying is enough
to fire a trade; the output reports exactly which one(s) did.

── Category 1: "direct" — Normal Category Call/Put ───────────────────────
  Category A signals (checked on CURRENT bar OR PREVIOUS bar, counted once):
      - Velocity Powered Buy/Sell
      - Acceleration Buy/Sell
      - CISD Dual Trailer flip (T1 or T2)
  Category B signals (checked on CURRENT bar only, no gate):
      - MVT Normal Buy/Sell
      - MVT Reverse Buy/Sell
      - EPSR Bull/Bear Thrust
      - SuperTrend Buy/Sell
      - MoVol-gated bar colour (state-based: true for every bar it holds
        that gated colour, not just the flip bar)
  Total count (max 8 per side) >= threshold (default 4) -> qualifies.

── Category 2: "fallback" — Continuation ──────────────────────────────────
  Fires only when the PREVIOUS bar's Category-1 count already met the
  threshold, AND the CURRENT bar's Category-1 count falls short of
  threshold (0 up to threshold-1), BUT PRT line and pi-breakout are still
  on the correct side of zero on the current bar (PRT>0 & pi_breakout>0
  for buy; PRT<0 & pi_breakout<0 for sell) -- read as "the move hasn't
  reversed, it's just quiet."

── Category 3: "regime_confirmation" — HMM Regime Confirmation ───────────
  All of the following must hold together (an AND, not a count):
    Buy:
      1. MVT HMM Regime = Bull Extreme ("Green") on the current bar AND
         the previous 2 bars (3 consecutive bars, strictly Bull Extreme --
         Bull Strong does NOT qualify)
      2. PRT line > 0 on the current bar
      3. pi-breakout > 0 on the current bar
      4. EPSR "not Red": epsr_norm >= 0 AND epsr_norm > epsr_norm[1] (rising)
    Sell (mirrored):
      1. MVT HMM Regime = Bear Extreme ("Red") on current bar + previous 2
         bars (Bear Strong does NOT qualify)
      2. PRT line < 0
      3. pi-breakout < 0
      4. EPSR "not Green": epsr_norm <= 0 AND epsr_norm < epsr_norm[1] (falling)

── Tie-break ───────────────────────────────────────────────────────────────
  If BOTH buy and sell qualify on the same bar (conflicting signal), the
  side with the higher Category-1 raw count wins; a true tie fires nothing
  (contradictory signal, sit it out).

── De-dup ────────────────────────────────────────────────────────────────
  A new decision only fires once per (bar_index, side) -- re-checking an
  already-fired bar/side on a later tick of the same bar does not re-fire,
  even if the set of contributing categories grows or shrinks.
"""

from dataclasses import dataclass, field


DEFAULT_ENABLED_SIGNALS = {
    "normal": True,
    "reverse": True,
    "velocity": True,
    "acceleration": True,
    "epsr_thrust": True,
    "supertrend": True,
    "movol_gated": True,
    "cisd_flip": True,
}


@dataclass
class StrategyConfig:
    threshold: int = 4
    enabled_signals: dict = field(default_factory=lambda: dict(DEFAULT_ENABLED_SIGNALS))
    enable_fallback: bool = True
    enable_regime_confirmation: bool = True


class SignalEngine:
    def __init__(self, cfg: StrategyConfig = None):
        self.cfg = cfg or StrategyConfig()
        self._last_fired = None  # (bar_index, side) of last fired trade
        self._prev_buy_count = None
        self._prev_sell_count = None
        self._last_bar_index = None

    # ── Category 1: raw signal count ──────────────────────────────────────
    def _raw_side_count(self, row, side: str):
        """Plain count, no gate. Returns (count, list of contributing signal names)."""
        en = self.cfg.enabled_signals
        hits = []

        if en["normal"] and row["final_full_buy" if side == "buy" else "final_full_sell"]:
            hits.append("mvt_normal")
        if en["reverse"] and row["longrvs" if side == "buy" else "shortrvs"]:
            hits.append("mvt_reverse")
        if en["epsr_thrust"] and row["bull_thrust" if side == "buy" else "bear_thrust"]:
            hits.append("epsr_thrust")
        if en["supertrend"] and row["st_signal_buy" if side == "buy" else "st_signal_sell"]:
            hits.append("supertrend")
        if en["movol_gated"]:
            gated = row["movol_gate_dir"]  # +1 bull, -1 bear, 0 none
            if (side == "buy" and gated == 1) or (side == "sell" and gated == -1):
                hits.append("movol_gated")

        if en["velocity"]:
            cur = row["velbuy"] if side == "buy" else row["velsell"]
            prev = row["velbuy_prev"] if side == "buy" else row["velsell_prev"]
            if cur or prev:
                hits.append("velocity")

        if en["acceleration"]:
            cur = row["acc_up"] if side == "buy" else row["acc_down"]
            prev = row["acc_up_prev"] if side == "buy" else row["acc_down_prev"]
            if cur or prev:
                hits.append("acceleration")

        if en["cisd_flip"]:
            if side == "buy":
                cur = row["cisd_bull_flip1"] or row["cisd_bull_flip2"]
                prev = row["cisd_bull_flip1_prev"] or row["cisd_bull_flip2_prev"]
            else:
                cur = row["cisd_bear_flip1"] or row["cisd_bear_flip2"]
                prev = row["cisd_bear_flip1_prev"] or row["cisd_bear_flip2_prev"]
            if cur or prev:
                hits.append("cisd_flip")

        return len(hits), hits

    # ── Category 2: fallback continuation ─────────────────────────────────
    def _fallback_qualifies(self, row, count, prev_count, side: str) -> bool:
        if not self.cfg.enable_fallback:
            return False
        th = self.cfg.threshold
        if prev_count is None or prev_count < th or count >= th:
            return False
        gate_ok = (row["prt_osc"] > 0 and row["pi_breakout"] > 0) if side == "buy" else \
                  (row["prt_osc"] < 0 and row["pi_breakout"] < 0)
        return gate_ok

    # ── Category 3: HMM regime confirmation ────────────────────────────────
    def _regime_confirmation_qualifies(self, row, side: str) -> bool:
        if not self.cfg.enable_regime_confirmation:
            return False
        try:
            dom0 = row["mvt_dom_code"]
            dom1 = row["mvt_dom_code_prev1"]
            dom2 = row["mvt_dom_code_prev2"]
            epsr = row["epsr_norm"]
            epsr_prev = row["epsr_norm_prev"]
            prt = row["prt_osc"]
            pib = row["pi_breakout"]
        except KeyError:
            return False

        if any(v is None for v in (dom0, dom1, dom2, epsr, epsr_prev, prt, pib)):
            return False
        import math
        if any(isinstance(v, float) and math.isnan(v) for v in (dom0, dom1, dom2, epsr, epsr_prev, prt, pib)):
            return False

        if side == "buy":
            regime_ok = (dom0 == 2) and (dom1 == 2) and (dom2 == 2)  # 3 consecutive Bull Extreme
            prt_ok = prt > 0
            pib_ok = pib > 0
            epsr_ok = (epsr >= 0) and (epsr > epsr_prev)
        else:
            regime_ok = (dom0 == -2) and (dom1 == -2) and (dom2 == -2)  # 3 consecutive Bear Extreme
            prt_ok = prt < 0
            pib_ok = pib < 0
            epsr_ok = (epsr <= 0) and (epsr < epsr_prev)

        return regime_ok and prt_ok and pib_ok and epsr_ok

    def _side_evaluation(self, row, prev_count, side: str):
        """
        Returns (qualifies: bool, count: int, hits: list, categories: list[str])
        categories lists every category ('direct' / 'fallback' /
        'regime_confirmation') that independently qualifies this bar/side.
        """
        count, hits = self._raw_side_count(row, side)
        th = self.cfg.threshold
        categories = []

        if count >= th:
            categories.append("direct")
        if self._fallback_qualifies(row, count, prev_count, side):
            categories.append("fallback")
        if self._regime_confirmation_qualifies(row, side):
            categories.append("regime_confirmation")

        return len(categories) > 0, count, hits, categories

    def evaluate_bar(self, row, bar_index):
        """
        bar_index: identifier for the bar being evaluated. In backtest this
        is the DataFrame row index (one call per closed bar). In live
        trading this should be the CURRENT (possibly still-forming) bar's
        index/timestamp -- call this once per incoming tick; de-dup below
        prevents re-firing on every tick of the same bar/side.

        Returns dict: {decision, buy_count, sell_count, buy_hits, sell_hits,
                        buy_categories, sell_categories}
        """
        buy_qual, buy_count, buy_hits, buy_categories = self._side_evaluation(row, self._prev_buy_count, "buy")
        sell_qual, sell_count, sell_hits, sell_categories = self._side_evaluation(row, self._prev_sell_count, "sell")

        decision = None
        winner = None
        if buy_qual and sell_qual:
            if buy_count > sell_count:
                winner = "buy"
            elif sell_count > buy_count:
                winner = "sell"
            # true tie -> no trade, contradictory signal
        elif buy_qual:
            winner = "buy"
        elif sell_qual:
            winner = "sell"

        if winner == "buy":
            sig = (bar_index, "buy")
            if sig != self._last_fired:
                decision = "CALL"
                self._last_fired = sig
        elif winner == "sell":
            sig = (bar_index, "sell")
            if sig != self._last_fired:
                decision = "PUT"
                self._last_fired = sig

        if bar_index != self._last_bar_index:
            self._prev_buy_count = buy_count
            self._prev_sell_count = sell_count
            self._last_bar_index = bar_index

        return {
            "decision": decision,
            "buy_count": buy_count, "sell_count": sell_count,
            "buy_hits": buy_hits, "sell_hits": sell_hits,
            "buy_categories": buy_categories, "sell_categories": sell_categories,
        }


def build_movol_gate_column(mvt_dom_codes, movol_dom_dirs):
    """
    mvt_dom_codes: list of int, +2/+1/0/-1/-2 (MVT HMM dominant state per bar)
    movol_dom_dirs: list of int, +1/0/-1 (MoVol HMM dominant direction per bar)
    Returns list of +1 (bull gated), -1 (bear gated), 0 (grey/no gate) per bar,
    mirroring the Pine "MoVol Gating (MVT+MoVol)" bar-colour logic.
    """
    out = []
    for mvt_dom, movol_dir in zip(mvt_dom_codes, movol_dom_dirs):
        if mvt_dom is None or movol_dir is None:
            out.append(0)
            continue
        if movol_dir > 0 and mvt_dom in (2, 1):
            out.append(1)
        elif movol_dir < 0 and mvt_dom in (-2, -1):
            out.append(-1)
        else:
            out.append(0)
    return out
