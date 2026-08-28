"""
S&P 500 Momentum + Growth + Quality Screener, with Backtesting
-----------------------------------------------------------------
Ranks S&P 500 stocks using a transparent, rules-based scoring model to surface
candidates that have historically-favorable setups for large moves — NOT a
prediction engine. See the in-app disclaimer.

Run with:
    streamlit run app.py
"""

import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")

st.set_page_config(page_title="S&P 500 Breakout Screener", layout="wide")

# --------------------------------------------------------------------------
# 1. Universe: pull current S&P 500 constituents from Wikipedia
# --------------------------------------------------------------------------

@st.cache_data(ttl=60 * 60 * 24)
def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    html = requests.get(url, headers=headers, timeout=15).text
    tables = pd.read_html(html)
    df = tables[0]
    df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)  # BRK.B -> BRK-B
    return df[["Symbol", "Security", "GICS Sector"]].rename(
        columns={"Security": "Name", "GICS Sector": "Sector"}
    )


# --------------------------------------------------------------------------
# 2. Technical feature helpers (pure price/volume — reusable for live + backtest)
# --------------------------------------------------------------------------

def compute_technical_features(hist: pd.DataFrame) -> dict:
    """Compute momentum/technical features using ONLY the price history handed
    in. Used both for the live screener (full history) and the backtester
    (history truncated to a historical 'as of' date, so there's no look-ahead)."""
    close = hist["Close"]
    vol = hist["Volume"]

    def pct_change_over(days):
        if len(close) <= days:
            return np.nan
        return (close.iloc[-1] / close.iloc[-days] - 1) * 100

    ret_1m = pct_change_over(21)
    ret_3m = pct_change_over(63)
    ret_6m = pct_change_over(126)
    ret_12m = pct_change_over(252)

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi_last = rsi.iloc[-1] if not rsi.empty else np.nan

    sma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else np.nan
    sma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else np.nan
    price = close.iloc[-1]

    avg_vol_30 = vol.tail(30).mean()
    avg_vol_90 = vol.tail(90).mean()
    vol_trend = (avg_vol_30 / avg_vol_90 - 1) * 100 if avg_vol_90 else np.nan

    return dict(
        price=price,
        ret_1m=ret_1m, ret_3m=ret_3m, ret_6m=ret_6m, ret_12m=ret_12m,
        rsi=rsi_last,
        above_sma50=(price > sma50) if pd.notna(sma50) else np.nan,
        above_sma200=(price > sma200) if pd.notna(sma200) else np.nan,
        vol_trend=vol_trend,
    )


def technical_score(df: pd.DataFrame) -> pd.Series:
    """The pure price/volume momentum score. This is the ONLY component that
    can be honestly backtested with free data, because analyst estimates and
    fundamentals from arbitrary past dates aren't available via this API."""
    z_ret_3m = zscore(df["ret_3m"])
    z_ret_6m = zscore(df["ret_6m"])
    z_vol_trend = zscore(df["vol_trend"])
    rsi_score = df["rsi"].apply(
        lambda x: 1.0 if pd.notna(x) and 45 <= x <= 65
        else (0.3 if pd.notna(x) and 30 <= x <= 75 else -0.5)
    )
    trend_score = (
        df["above_sma50"].fillna(False).astype(int)
        + df["above_sma200"].fillna(False).astype(int)
    ) - 1
    return (
        0.45 * z_ret_3m
        + 0.25 * z_ret_6m
        + 0.15 * z_vol_trend
        + 0.10 * rsi_score
        + 0.05 * trend_score
    )


# --------------------------------------------------------------------------
# 3. Live data pull per ticker (price history + fundamentals + analyst data)
# --------------------------------------------------------------------------

@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def fetch_ticker_data(ticker: str, include_upgrade_momentum: bool = False):
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period="14mo", interval="1d", auto_adjust=True)
        if hist.empty or len(hist) < 200:
            return None

        info = tk.info or {}
        tech = compute_technical_features(hist)

        target_mean = info.get("targetMeanPrice")
        price = tech["price"]
        analyst_upside = (
            (target_mean / price - 1) * 100 if target_mean and price else np.nan
        )

        # --- Fundamentals (quality) ---
        roe = info.get("returnOnEquity")
        profit_margin = info.get("profitMargins")
        debt_to_equity = info.get("debtToEquity")
        current_ratio = info.get("currentRatio")
        earnings_growth = info.get("earningsGrowth")
        revenue_growth = info.get("revenueGrowth")
        peg = info.get("pegRatio")
        forward_pe = info.get("forwardPE")

        # --- Analyst ratings ---
        rec_key = info.get("recommendationKey", "n/a")
        rec_mean = info.get("recommendationMean")  # 1=Strong Buy ... 5=Sell
        num_analysts = info.get("numberOfAnalystOpinions", np.nan)

        upgrade_momentum = np.nan
        if include_upgrade_momentum:
            try:
                recs = tk.get_recommendations_summary() if hasattr(tk, "get_recommendations_summary") else None
                # Fall back to upgrades/downgrades history if available
                ud = tk.upgrades_downgrades
                if ud is not None and not ud.empty:
                    cutoff = pd.Timestamp.now(tz=ud.index.tz) - pd.Timedelta(days=90)
                    recent = ud[ud.index >= cutoff]
                    up = recent["ToGrade"].str.contains(
                        "Buy|Outperform|Overweight", case=False, na=False
                    ).sum()
                    down = recent["ToGrade"].str.contains(
                        "Sell|Underperform|Underweight", case=False, na=False
                    ).sum()
                    upgrade_momentum = up - down
            except Exception:
                upgrade_momentum = np.nan

        beta = info.get("beta")
        market_cap = info.get("marketCap")
        sector = info.get("sector", "")
        short_pct_float = info.get("shortPercentOfFloat")

        return dict(
            ticker=ticker,
            **tech,
            analyst_upside=analyst_upside,
            rec_key=rec_key,
            rec_mean=rec_mean,
            num_analysts=num_analysts,
            upgrade_momentum=upgrade_momentum,
            roe=(roe * 100 if roe else np.nan),
            profit_margin=(profit_margin * 100 if profit_margin else np.nan),
            debt_to_equity=debt_to_equity,
            current_ratio=current_ratio,
            earnings_growth=(earnings_growth * 100 if earnings_growth else np.nan),
            revenue_growth=(revenue_growth * 100 if revenue_growth else np.nan),
            peg=peg,
            forward_pe=forward_pe,
            beta=beta,
            market_cap=market_cap,
            sector=sector,
            short_pct_float=(short_pct_float * 100 if short_pct_float else np.nan),
        )
    except Exception:
        return None


# --------------------------------------------------------------------------
# 4. Scoring (live screener — combines technicals + fundamentals + analysts)
# --------------------------------------------------------------------------

def zscore(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    mean, std = s.mean(skipna=True), s.std(skipna=True)
    if not std or np.isnan(std):
        return pd.Series(0, index=s.index)
    return ((s - mean) / std).fillna(0)


def score_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ---- Short-term score (3-6mo horizon): momentum + technical + analyst sentiment
    z_ret_3m = zscore(df["ret_3m"])
    z_ret_6m = zscore(df["ret_6m"])
    z_vol_trend = zscore(df["vol_trend"])
    z_analyst_upside = zscore(df["analyst_upside"])
    z_rec_mean_inv = -zscore(df["rec_mean"])          # lower rec_mean = more bullish
    z_upgrade_momentum = zscore(df["upgrade_momentum"])
    rsi_score = df["rsi"].apply(
        lambda x: 1.0 if pd.notna(x) and 45 <= x <= 65
        else (0.3 if pd.notna(x) and 30 <= x <= 75 else -0.5)
    )
    trend_score = (
        df["above_sma50"].fillna(False).astype(int)
        + df["above_sma200"].fillna(False).astype(int)
    ) - 1

    has_upgrade_data = df["upgrade_momentum"].notna().any()
    if has_upgrade_data:
        df["short_term_score"] = (
            0.25 * z_ret_3m
            + 0.15 * z_ret_6m
            + 0.10 * z_vol_trend
            + 0.20 * z_analyst_upside
            + 0.10 * z_rec_mean_inv
            + 0.10 * z_upgrade_momentum
            + 0.07 * rsi_score
            + 0.03 * trend_score
        )
    else:
        df["short_term_score"] = (
            0.28 * z_ret_3m
            + 0.17 * z_ret_6m
            + 0.12 * z_vol_trend
            + 0.23 * z_analyst_upside
            + 0.12 * z_rec_mean_inv
            + 0.08 * rsi_score
        )

    # ---- Long-term score (12mo horizon): growth + valuation + quality + analysts
    z_earnings_growth = zscore(df["earnings_growth"])
    z_revenue_growth = zscore(df["revenue_growth"])
    z_ret_12m = zscore(df["ret_12m"])
    z_fwd_pe_inv = -zscore(df["forward_pe"])
    z_peg_inv = -zscore(df["peg"])
    z_roe = zscore(df["roe"])
    z_profit_margin = zscore(df["profit_margin"])
    z_debt_to_equity_inv = -zscore(df["debt_to_equity"])
    z_analyst_upside_lt = zscore(df["analyst_upside"])
    z_rec_mean_inv_lt = -zscore(df["rec_mean"])

    df["long_term_score"] = (
        0.18 * z_earnings_growth
        + 0.14 * z_revenue_growth
        + 0.16 * z_analyst_upside_lt
        + 0.10 * z_rec_mean_inv_lt
        + 0.10 * z_ret_12m
        + 0.09 * z_fwd_pe_inv
        + 0.08 * z_peg_inv
        + 0.08 * z_roe
        + 0.04 * z_profit_margin
        + 0.03 * z_debt_to_equity_inv
    )

    return df


# --------------------------------------------------------------------------
# 5. Backtest engine (technical/momentum component ONLY — see disclaimer)
# --------------------------------------------------------------------------

LOOKBACK_TRADING_DAYS = {"3 months": 63, "6 months": 126, "12 months": 252}


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def fetch_backtest_history(ticker: str, period: str):
    try:
        hist = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
        if hist.empty:
            return None
        return hist
    except Exception:
        return None


def run_backtest(tickers, lookback_label, benchmark_symbol="^GSPC"):
    lookback_days = LOOKBACK_TRADING_DAYS[lookback_label]
    period = "36mo"  # generous window: enough history before AND after the as-of date

    rows = []
    for t in tickers:
        hist = fetch_backtest_history(t, period)
        if hist is None or len(hist) < lookback_days + 210:
            continue  # not enough history before the as-of point for 200SMA + after it for forward return

        as_of_idx = len(hist) - 1 - lookback_days
        if as_of_idx < 210:
            continue

        hist_as_of = hist.iloc[: as_of_idx + 1]
        feats = compute_technical_features(hist_as_of)

        price_as_of = hist["Close"].iloc[as_of_idx]
        price_now = hist["Close"].iloc[-1]
        forward_return = (price_now / price_as_of - 1) * 100

        feats["ticker"] = t
        feats["as_of_date"] = hist.index[as_of_idx].date()
        feats["forward_return"] = forward_return
        rows.append(feats)

    if not rows:
        return None, None

    df = pd.DataFrame(rows)
    df["momentum_score"] = technical_score(df)

    # Benchmark
    bench_hist = fetch_backtest_history(benchmark_symbol, period)
    bench_return = np.nan
    if bench_hist is not None and len(bench_hist) > lookback_days:
        bench_as_of_idx = len(bench_hist) - 1 - lookback_days
        bench_return = (
            bench_hist["Close"].iloc[-1] / bench_hist["Close"].iloc[bench_as_of_idx] - 1
        ) * 100

    return df, bench_return


# --------------------------------------------------------------------------
# 6. Streamlit UI
# --------------------------------------------------------------------------

st.title("📈 S&P 500 Momentum, Growth & Quality Screener")

st.warning(
    "**This is not financial advice.** This tool ranks stocks using public "
    "price, technical, fundamental, and analyst-estimate data. It cannot "
    "predict specific price moves like '+30% in 6 months.' Momentum, "
    "analyst targets, and past backtest results are not guarantees of "
    "future returns. Always do your own research and consider consulting a "
    "licensed financial advisor before investing."
)

mode = st.radio("Mode", ["🔍 Live Screener", "🕰️ Backtest"], horizontal=True)

# ============================== LIVE SCREENER ==============================
if mode == "🔍 Live Screener":
    with st.sidebar:
        st.header("Screener Settings")
        sample_size = st.slider(
            "Number of S&P 500 stocks to scan (larger = slower)",
            min_value=50, max_value=503, value=150, step=25,
            help="Scanning all 503 names can take several minutes on first run "
                 "due to API rate limits. Cached results refresh every 6 hours."
        )
        min_market_cap_b = st.number_input(
            "Minimum market cap ($B)", min_value=0, max_value=500, value=2
        )
        include_upgrades = st.checkbox(
            "Include analyst upgrade/downgrade momentum (slower — extra API call per stock)",
            value=False,
        )
        run_btn = st.button("🔍 Run Screener", type="primary")

        st.divider()
        st.caption(
            "**Short-term (3-6mo)** weights price momentum, rising volume, "
            "analyst target upside, analyst rating (and recent upgrades if "
            "enabled), and RSI positioning.\n\n"
            "**Long-term (12mo)** weights earnings/revenue growth, analyst "
            "target upside & rating, valuation (forward P/E, PEG), and "
            "quality (ROE, profit margin, debt/equity)."
        )

    if run_btn:
        tickers_df = get_sp500_tickers()
        universe = tickers_df.sample(n=min(sample_size, len(tickers_df)), random_state=42) \
            if sample_size < len(tickers_df) else tickers_df

        progress = st.progress(0, text="Fetching data...")
        results = []
        tickers_list = universe["Symbol"].tolist()

        for i, t in enumerate(tickers_list):
            data = fetch_ticker_data(t, include_upgrade_momentum=include_upgrades)
            if data:
                results.append(data)
            progress.progress((i + 1) / len(tickers_list), text=f"Fetched {t} ({i+1}/{len(tickers_list)})")

        progress.empty()

        if not results:
            st.error("No data could be fetched. Check your internet connection / API limits.")
            st.stop()

        df = pd.DataFrame(results)
        df = df.merge(tickers_df, left_on="ticker", right_on="Symbol", how="left")
        df = df[df["market_cap"].fillna(0) >= min_market_cap_b * 1e9]
        df = df.dropna(subset=["ret_3m", "ret_12m"], how="all")

        if df.empty:
            st.error("No stocks passed the filters. Try lowering the market cap filter.")
            st.stop()

        scored = score_dataframe(df)
        st.session_state["scored"] = scored
        st.session_state["scan_time"] = datetime.now().strftime("%Y-%m-%d %H:%M")

    if "scored" in st.session_state:
        scored = st.session_state["scored"]
        st.caption(f"Last scan: {st.session_state['scan_time']} • {len(scored)} stocks analyzed")

        tab1, tab2, tab3 = st.tabs(
            ["🚀 Short-Term (3-6mo) Top 5", "📊 Long-Term (12mo) Top 5", "🔍 Full Data"]
        )

        display_cols_short = [
            "ticker", "Name", "Sector", "price", "ret_3m", "ret_6m",
            "rsi", "analyst_upside", "rec_key", "num_analysts", "short_term_score",
        ]
        display_cols_long = [
            "ticker", "Name", "Sector", "price", "ret_12m", "earnings_growth",
            "revenue_growth", "forward_pe", "peg", "roe", "analyst_upside",
            "rec_key", "long_term_score",
        ]

        with tab1:
            st.subheader("Top 5 short-term momentum candidates")
            top_short = scored.sort_values("short_term_score", ascending=False).head(5)
            st.dataframe(
                top_short[display_cols_short].style.format({
                    "price": "${:.2f}", "ret_3m": "{:.1f}%", "ret_6m": "{:.1f}%",
                    "rsi": "{:.0f}", "analyst_upside": "{:.1f}%", "short_term_score": "{:.2f}",
                }),
                use_container_width=True, hide_index=True,
            )
            for _, row in top_short.iterrows():
                with st.expander(f"{row['ticker']} — {row['Name']}"):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("3mo return", f"{row['ret_3m']:.1f}%")
                    c2.metric("6mo return", f"{row['ret_6m']:.1f}%")
                    c3.metric("RSI (14)", f"{row['rsi']:.0f}" if pd.notna(row['rsi']) else "n/a")
                    c4.metric("Analyst upside to target", f"{row['analyst_upside']:.1f}%" if pd.notna(row['analyst_upside']) else "n/a")
                    c5, c6, c7 = st.columns(3)
                    c5.metric("Analyst rating", row['rec_key'] if pd.notna(row['rec_key']) else "n/a")
                    c6.metric("# Analysts", f"{row['num_analysts']:.0f}" if pd.notna(row['num_analysts']) else "n/a")
                    if pd.notna(row.get('upgrade_momentum')):
                        c7.metric("Upgrades - downgrades (90d)", f"{row['upgrade_momentum']:.0f}")
                    st.caption(f"Above 50-day MA: {row['above_sma50']} | Above 200-day MA: {row['above_sma200']}")

        with tab2:
            st.subheader("Top 5 long-term growth candidates")
            top_long = scored.sort_values("long_term_score", ascending=False).head(5)
            st.dataframe(
                top_long[display_cols_long].style.format({
                    "price": "${:.2f}", "ret_12m": "{:.1f}%", "earnings_growth": "{:.1f}%",
                    "revenue_growth": "{:.1f}%", "forward_pe": "{:.1f}", "peg": "{:.2f}",
                    "roe": "{:.1f}%", "analyst_upside": "{:.1f}%", "long_term_score": "{:.2f}",
                }),
                use_container_width=True, hide_index=True,
            )
            for _, row in top_long.iterrows():
                with st.expander(f"{row['ticker']} — {row['Name']}"):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("12mo return", f"{row['ret_12m']:.1f}%")
                    c2.metric("Earnings growth est.", f"{row['earnings_growth']:.1f}%" if pd.notna(row['earnings_growth']) else "n/a")
                    c3.metric("Forward P/E", f"{row['forward_pe']:.1f}" if pd.notna(row['forward_pe']) else "n/a")
                    c4.metric("Analyst upside to target", f"{row['analyst_upside']:.1f}%" if pd.notna(row['analyst_upside']) else "n/a")
                    c5, c6, c7 = st.columns(3)
                    c5.metric("ROE", f"{row['roe']:.1f}%" if pd.notna(row['roe']) else "n/a")
                    c6.metric("Debt/Equity", f"{row['debt_to_equity']:.0f}" if pd.notna(row['debt_to_equity']) else "n/a")
                    c7.metric("Analyst rating", row['rec_key'] if pd.notna(row['rec_key']) else "n/a")
                    st.caption(f"Revenue growth est.: {row['revenue_growth']:.1f}% | PEG: {row['peg']}" if pd.notna(row['revenue_growth']) else "")

        with tab3:
            st.subheader("All scanned stocks")
            sort_col = st.selectbox("Sort by", ["short_term_score", "long_term_score", "ret_3m", "ret_12m", "roe", "market_cap"])
            st.dataframe(
                scored.sort_values(sort_col, ascending=False),
                use_container_width=True, hide_index=True,
            )
            csv = scored.to_csv(index=False).encode("utf-8")
            st.download_button("Download full results as CSV", csv, "sp500_screen.csv", "text/csv")
    else:
        st.info("Set your options in the sidebar and click **Run Screener** to begin.")

# ================================ BACKTEST =================================
else:
    st.subheader("🕰️ Backtest the momentum strategy")
    st.info(
        "**Scope of this backtest:** it can only replay the *technical/momentum* "
        "part of the strategy (price returns, RSI, volume trend, moving averages) "
        "because free data sources don't provide point-in-time snapshots of "
        "analyst estimates or fundamentals from arbitrary past dates — only "
        "today's values. The live screener's fundamentals/analyst weighting is "
        "therefore NOT reflected here. Treat this as a check on the momentum "
        "signal specifically, not the full strategy.",
        icon="ℹ️",
    )

    with st.sidebar:
        st.header("Backtest Settings")
        bt_sample_size = st.slider(
            "Number of S&P 500 stocks to include", min_value=30, max_value=300,
            value=100, step=10,
        )
        bt_lookback = st.selectbox(
            "As-of date (how far back to rank stocks)",
            ["3 months", "6 months", "12 months"], index=1,
        )
        bt_run = st.button("▶️ Run Backtest", type="primary")

    if bt_run:
        tickers_df = get_sp500_tickers()
        universe = tickers_df.sample(n=min(bt_sample_size, len(tickers_df)), random_state=7) \
            if bt_sample_size < len(tickers_df) else tickers_df

        with st.spinner(f"Reconstructing momentum ranks as of {bt_lookback} ago and measuring what happened since..."):
            bt_df, bench_return = run_backtest(universe["Symbol"].tolist(), bt_lookback)

        if bt_df is None or bt_df.empty:
            st.error("Not enough historical data to run this backtest. Try a smaller lookback or different sample.")
            st.stop()

        bt_df = bt_df.merge(tickers_df, left_on="ticker", right_on="Symbol", how="left")
        st.session_state["bt_df"] = bt_df
        st.session_state["bt_bench"] = bench_return
        st.session_state["bt_lookback"] = bt_lookback

    if "bt_df" in st.session_state:
        bt_df = st.session_state["bt_df"]
        bench_return = st.session_state["bt_bench"]
        lookback_label = st.session_state["bt_lookback"]

        top5 = bt_df.sort_values("momentum_score", ascending=False).head(5)

        as_of_date = bt_df["as_of_date"].iloc[0]
        st.caption(
            f"Ranked {len(bt_df)} stocks as of **{as_of_date}** using only data "
            f"available up to that date, then measured actual returns from "
            f"{as_of_date} to today ({lookback_label} later)."
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Top 5 avg. forward return", f"{top5['forward_return'].mean():.1f}%")
        c2.metric("Full universe avg. forward return", f"{bt_df['forward_return'].mean():.1f}%")
        c3.metric("S&P 500 index (benchmark) return", f"{bench_return:.1f}%" if pd.notna(bench_return) else "n/a")

        hit_rate = (top5["forward_return"] >= 30).sum()
        st.write(f"**{hit_rate} of the top 5** actually returned +30% or more over that period.")

        st.subheader(f"Top 5 by momentum score as of {as_of_date}")
        st.dataframe(
            top5[["ticker", "Name", "Sector", "momentum_score", "ret_3m", "ret_6m", "rsi", "forward_return"]]
            .style.format({
                "momentum_score": "{:.2f}", "ret_3m": "{:.1f}%", "ret_6m": "{:.1f}%",
                "rsi": "{:.0f}", "forward_return": "{:.1f}%",
            })
            .background_gradient(subset=["forward_return"], cmap="RdYlGn"),
            use_container_width=True, hide_index=True,
        )

        st.subheader("All ranked stocks")
        st.dataframe(
            bt_df.sort_values("momentum_score", ascending=False)[
                ["ticker", "Name", "Sector", "momentum_score", "forward_return"]
            ],
            use_container_width=True, hide_index=True,
        )

        st.caption(
            "Reminder: one historical window is not statistically meaningful on "
            "its own — momentum strategies can and do have long losing streaks. "
            "Re-run with different lookback periods and sample sizes to see how "
            "consistent (or not) the edge really is."
        )
    else:
        st.info("Set your options in the sidebar and click **Run Backtest** to begin.")
