import numpy as np
import pandas as pd
import streamlit as st

from pipeline import run_pipeline

st.set_page_config(page_title="Trade Strategy Demo", page_icon="📈", layout="wide")


def generate_sample_data(n_bars: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    base_price = 100.0
    drift = np.linspace(0.0, 0.8, n_bars)
    noise = rng.normal(0.0, 0.35, size=n_bars)
    close = base_price + np.cumsum((drift + noise) / 100.0)
    open_ = np.empty(n_bars, dtype=float)
    open_[0] = close[0]
    open_[1:] = close[:-1] + rng.normal(0.0, 0.1, size=n_bars - 1)
    high = np.maximum(open_, close) + np.abs(rng.normal(0.0, 0.2, size=n_bars))
    low = np.minimum(open_, close) - np.abs(rng.normal(0.0, 0.2, size=n_bars))
    volume = rng.integers(1000, 4000, size=n_bars)

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


st.title("Trade Strategy Demo")
st.caption("Run your trading pipeline from a browser-based environment, including on a phone.")

n_bars = st.slider("How many synthetic bars to simulate?", 120, 600, 240, 60)

if st.button("Run strategy demo"):
    with st.spinner("Running the strategy pipeline..."):
        df = generate_sample_data(n_bars)
        results = run_pipeline(df, tf_cal="M5 Trained", seconds_per_bar=60)

    st.subheader("Recent decisions")
    decision_counts = results["decision"].dropna().value_counts().to_dict()
    st.write(decision_counts if decision_counts else {"No trades": 0})

    st.subheader("Latest rows")
    st.dataframe(
        results.tail(15)[["open", "high", "low", "close", "volume", "decision", "buy_count", "sell_count"]],
        use_container_width=True,
    )

    st.info("This demo uses synthetic OHLCV data so it can run in a browser environment even without live Deriv access.")
