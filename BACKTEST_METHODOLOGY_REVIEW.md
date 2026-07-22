# Backtest Methodology Review — Binary Options (frxEURUSD, 5m)

> Review of `backtest.py`, `config.py`, `checkpoint.py`, `pipeline.py`,
> `tick_replay.py`, and the in-progress checkpoint pair
> `backtest_checkpoints/{candles_5m_auto_M5.csv, checkpoint_5m_auto_M5.json}`.
> Written while the backtest was still running (Pass 2 not yet scoring trades).

---

## 1. Context — what the data shows so far

### What was on disk at review time

| Artifact | Value |
| --- | --- |
| **Config running** | `5m_auto_M5` (only one of the three in `BACKTEST_CONFIGS`) |
| **Symbol / TF** | frxEURUSD, 5-minute bars |
| **Target candles** | 5,000 (per `config.BACKTEST_HISTORY_CANDLES`) |
| **Candles actually pulled** | 3,777 (~76% of target — likely Deriv's free-tier history ceiling on forex) |
| **Window of data** | 2026‑06‑30 12:15 UTC → 2026‑07‑17 20:55 UTC (~17.4 days) |

### Candle CSV — price action context

A textbook **choppy, slightly bullish forex regime** — exactly where
range-bound strategies tend to struggle:

- Close: 1.13952 → 1.14379 (**+0.375%** net drift over 17 days)
- Up-bars: **47.9%**, Down-bars: **49.0%**, Flat: **3.1%** — basically 50/50
- Mean per-bar range: **2.65 pips**, peak spike: **51.2 pips** (a news drop)
- Read: market is **ranging, not trending**

### Checkpoint JSON — pipeline state

```
flagged bars:         729
processed (Pass 2):     0   ← not started yet
trades scored:          0
skipped bars:           0
dropped bars:           0
clusters captured:    258   ← all of them, capture stage finished
skipped clusters:       0
```

- ✅ Pass 1 (bar-close pipeline) finished — 729 bars flagged
- ✅ Cluster trajectory capture finished — all 258 clusters cached with raw + replayed per-second signal counts
- ⏳ Pass 2 (tick replay + trade scoring) just started — no trades yet

### Signal density and cluster shape

- ~**42 flagged bars per day** across all clusters
- Median gap between flagged bars: **15 min** (median 900 s, mean 2048 s)
- **Two-thirds** (486 / 728) of flagged bars sit within 5 bars of another flagged bar — the strategy is clustering, not firing random one-offs
- **243 clusters** by Pass‑2 adjacency rule; **258 trajectories** by direction-aware grouping — the gap (~15) reflects opposite-direction near-misses
- **74.5% of clusters are multi-bar runs** in one direction (largest = 18 bars back-to-back)
- Per-trajectory "signal firing" rate: **21.6% on average**, range 0.5%–66% — wide spread suggests some clusters are real, some are noise

### What can't be said yet

- Win rate (no trades scored)
- CALL vs PUT split (would require re-deriving the cached `decision` column)
- Per-duration win rate per combo / per category (final output of `build_results_tables()`)

---

## 2. Verdict

The methodology is **genuinely above-average for an indicator-driven binary
options backtest**, but it is still likely to overstate edge by a few
percentage points relative to live trading, and the dataset is way too
small to be statistically conclusive either way.

---

## 3. Strengths (methodology gets right)

1. **Two-pass entry timing (bar-close → tick replay).** Most homebrew binary
   backtests naively use the bar's close as the entry. You can't actually
   trade at the last tick of a bar — by the time it closes, the next bar
   has started. Pass 2 reproducing the *true* entry tick is the correct
   pessimistic adjustment.
2. **Tick-rule exit at `entry + duration`.** Not bar-mid, not bar-close.
3. **`TIE_COUNTS_AS_LOSS = True`** in `config.py` is **conservative** — gives
   you a *lower bound* on edge.
4. **Edge-of-data drop** (`drop_edge_of_data_bars` in `backtest.py`): bars
   that can't be fully resolved out to the longest tested duration are not
   scored.
5. **Per-cluster trajectory capture → repaint analysis**: catches the #1
   pitfall in indicator trading — features that wouldn't have been
   knowable at decision time. `repaint_analysis.py` running per-second
   qualifying-sample checks against captured ticks is unusually thoughtful.
6. **Atomic checkpoint writes** (write-tmp + `os.replace`): reproducible
   bit-for-bit on resume, no silent corruption on partial writes.
7. **HMM regime gating** (`MOVOL_MODEL = "S=4 (Dir. x Vol)"`): not a
   vanilla oscillator; gating entries on a learned regime prior.
8. **Same-direction consecutive-cluster trajectory grouping** is
   resource-aware — single ticks pulled per cluster rather than per bar.
9. **Three configs run** (`5m_auto_M5`, `15m_auto_M15`, `5m_mismatch_M15`):
   A/B comparison between matched and deliberately mismatched TF-cal
   pairs is the right diagnostic for whether the HMM generalization is
   real or an artifact.

---

## 4. Weaknesses (ranked by impact, most-likely-to-fool-you first)

### 4.1 Hidden payout-ratio assumption

**Severity: high.** Deriv Rise/Fall payouts are typically ~95% on the
*winning* side, not 100%.

- Breakeven win rate at 100% payout (your implicit model) = **50.00%**
- Breakeven win rate at 95% payout = **51.28%**

`DEFAULT_STAKE = 1.0` and `TIE_COUNTS_AS_LOSS = True` in `config.py`
provide no payout-ratio constant anywhere. With 1:1 implied payout, a
50.5% win-rate strategy *looks* +ev but is **marginal-to-losing live**
under realistic Deriv payout. Explicitly model payout, recompute EV.

### 4.2 Cluster autocorrelation masquerading as statistical edge

**Severity: high.** The single biggest hidden danger in binary backtesting.

- 729 scored bars → 243 clusters (or 258 with direction-split grouping)
- 74.5% are multi-bar runs; largest cluster = 18 bars
- Bar-level observations are NOT independent — they share HMM regime,
  drift, news shock, session
- Standard error `sqrt(p(1-p)/N)` with N=729 *vastly* underestimates true
  variance
- Effective sample size is probably ~150–250, not 729
- A backtest win rate of e.g. 56% with naive SE will look like p < 0.001
  but is probably closer to p ≈ 0.05 with 95% CI ±5–7 pp

**Fix:** bootstrap on **clusters** with replacement; or one observation
per HMM regime episode.

### 4.3 17 days is too short for regime coverage

**Severity: high.** 1,800 candles is ~12.5 days of forex trading with
weekends stripped. You need **6–12 months** spanning 2–3+ distinct macro
regimes.

Worse: this 17-day window (June 30 → July 17, 2026) is a **sideways
range**, the literal worst regime in which to validate a directional
strategy. Strategies that look great in trends can look marginally
profitable in ranges, and vice versa. **You may be fitting to a regime
that's currently friendly and would break in a Fed-week regime flip.**

### 4.4 No walk-forward validation

**Severity: high.** Pure in-sample fit = unsurprising fit. Without
out-of-sample holdout, you can't distinguish:

- "the HMM caught real regime structure" from
- "the HMM memorized a 17-day window"

**Fix:** rolling-window 4-week train + 2-week test, repeat over 6+
windows, report mean + std of out-of-sample win rate by combo/duration.

### 4.5 No commission / spread / slippage model on forex

**Severity: medium-high (for forex; low if you switch to R_100).**

- frxEURUSD has real spread (~0.5–1.0 pip on Deriv)
- 2.65-pip average bar range makes spread non-trivial relative to
  expected move
- Tick-replay picks the actual tick that fires the signal, but in live
  trading you have spread between bid/ask

**Fix:** either model the spread explicitly, or switch to `R_100`
(Volatility 100 Index, zero-spread synthetic — already toggleable in
`config.py`) for clean signal-vs-noise evaluation first.

### 4.6 No drawdown / streak / risk-of-ruin analysis

**Severity: medium.** Binary options are unusually vulnerable to streak
risk. A 55% win-rate strategy can still wipe a fixed-stake bankroll if its
loss streaks cluster (which they will if the strategy is regime-driven —
when the HMM exits the favorable regime, you get 10 losses in a row).
`DEFAULT_STAKE = 1.0` sizes are not informed by any streak-distribution
analysis.

### 4.7 `SIGNAL_THRESHOLD = 4` is hard-coded, no sweep

**Severity: medium.** If you tune this, do it inside walk-forward folds,
never on the full sample. Otherwise you're peeking. Same logic applies
to `MOVOL_LOOKBACK_HOURS = 3.0` and `MOVOL_MODEL = "S=4 (Dir. x Vol)"`.

### 4.8 Repaint analysis is post-hoc, not adversarial

**Severity: low-medium.** Good: repaint_analysis.py checks per-second
qualifying-sample stability as the bar develops. **Missing:** cross-
parameter repaint detection — "would the signal have fired with a
different threshold or earlier max-bars lookback?" That's the canonical
parameter-repaint trap; worth adding.

### 4.9 Dropped-bar count = 0 across 729 flagged bars

**Severity: informational.** Probably correct (last 10 min of chart had
no flagged bars), but the `drop_edge_of_data_bars` call only fires if
there are candidates. Sanity-check by ensuring the max duration (10 min)
plus 30s settlement buffer leaves room for the very last flagged bar.

---

## 5. The math the backtest should be reporting (and isn't)

Each **(duration, combo, category)** row should report *expected value per
contract*, not just win rate:

```
EV_per_dollar = payout * win_rate - 1 * loss_rate - 1 * tie_rate
              = payout * win_rate - (1 - win_rate)
              = (payout + 1) * win_rate - 1
```

With `stake = 1.0`, `payout ≈ 1.95` (typical Deriv), this reduces to:

```
EV ≈ +0.95 * win_rate - 1.05 * loss_rate
```

- Win rate ≥ **51.28%** → positive EV under 95% payout
- Win rate < **51.28%** → negative; you'd *prefer* fewer high-confidence
  trades over many ~50% trades

`build_results_tables()` in `backtest.py` outputs win rate only. **The
risk of mistaking a 52% win-rate strategy for an edge over 50% breakeven,
when you correctly modeled 95% payout, is non-trivial.**

---

## 6. Pre-live changes required

1. Add `PAYOUT_RATIO = 0.95` to `config.py`.
2. Modify `build_results_tables()` to compute and report
   `expected_value_per_dollar` per row, output alongside win rate.
3. Bootstrap cluster-level win rates for 95% CIs (not per-bar).
4. Pull ≥6 months of history; cover 3+ regimes (range, trend, news-spike).
5. Implement a parameter-loss curve / max-loss-per-hour stop; test
   historical max drawdowns against that limit.
6. Switch to `R_100` (synthetic, zero-spread) before accepting the
   forex-spread hit in live.
7. Add cross-parameter repaint detection to `repaint_analysis.py`.

---

## 7. Bottom line

Real signal model + real entry/exit realism + real repaint awareness —
**but a dataset that's almost certainly too small and too single-regime
to be statistically meaningful.**

Estimated probability this strategy is +ev live: **40–60%**.

**Treat current results as a hypothesis, not a verdict.**
