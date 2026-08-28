# S&P 500 Momentum & Growth Screener

A Streamlit app that screens S&P 500 stocks and surfaces the top 5 candidates
for a **short-term (3–6 month)** setup and the top 5 for a **long-term
(12 month)** setup, based on a transparent, rules-based scoring model.

## ⚠️ Important — what this tool is and isn't

This is a **screener**, not a prediction engine. No tool — including
professional hedge fund models — can reliably tell you which specific stocks
will rise more than 30% in a given window. What this app *does* do:

- Pulls live price history and fundamentals for S&P 500 companies (via Yahoo Finance)
- Scores stocks on momentum, technical positioning, analyst price targets,
  and growth/valuation metrics
- Ranks them so you can see which names currently have the most favorable
  combination of those factors

Treat the output as a **starting point for your own research**, not a
recommendation. Past momentum and analyst optimism are not guarantees of
future returns, and analyst price targets are frequently wrong. Consider
talking to a licensed financial advisor before making investment decisions.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

## How to use it

1. In the sidebar, choose how many S&P 500 stocks to scan (start with ~100–150;
   scanning all 503 takes a few minutes due to Yahoo Finance rate limits).
2. Set a minimum market cap filter if you want to exclude smaller names.
3. Click **Run Screener**.
4. View results in the **Short-Term** and **Long-Term** tabs, or browse/download
   the full dataset in **Full Data**.

Results are cached for 6 hours so re-running with the same settings is fast.

## Methodology detail

**Short-term score** (weighted composite, z-scored across the scanned universe):
- Price momentum — 3-month and 6-month returns
- 30-day vs 90-day average volume trend (rising volume = building interest)
- Upside to average analyst price target
- Analyst rating (recommendation mean, Strong Buy → Sell scale)
- Optional: net analyst upgrades minus downgrades in the trailing 90 days
- RSI(14) positioned in a healthy continuation zone (45–65)
- Price above 50-day and 200-day moving averages

**Long-term score**:
- Forward earnings and revenue growth estimates
- Upside to average analyst price target, and analyst rating
- 12-month price return
- Valuation: forward P/E and PEG ratio (lower is better, inverted)
- Quality: ROE, profit margin, debt/equity (lower is better, inverted)

Exact weights are in `score_dataframe()` in `app.py` — plain Python, easy to
tune.

## Backtest mode

The **Backtest** tab tests whether the momentum signal actually worked
historically. It:

1. Picks an "as of" date N months in the past (your choice: 3/6/12 months ago).
2. Computes the **technical/momentum score using only price and volume data
   available up to that date** — no look-ahead.
3. Takes the top 5 stocks by that historical score.
4. Reports what those 5 stocks *actually* returned from that date to today,
   compared against the full scanned universe and the S&P 500 index (`^GSPC`).

**Important limitation:** this only backtests the momentum/technical portion
of the strategy. Point-in-time analyst estimates, price targets, and
fundamentals from arbitrary past dates aren't available through free APIs
like Yahoo Finance — only current snapshots are. So the live screener's
fundamentals + analyst-rating weighting can't be backtested the same
rigorous way, and the backtest numbers will not exactly predict how the live
screener's combined score would have performed. Treat the backtest as a
sanity check on the momentum piece, not a validation of the whole tool.

A single historical window is also not statistically meaningful by itself —
try several different lookback periods and sample sizes before drawing
conclusions about whether the approach has a real edge.

## Data source & limitations

- Data comes from Yahoo Finance via the `yfinance` library. It's free but
  can be delayed, occasionally incomplete for certain tickers, or rate-limited
  if you scan too many stocks too quickly. Enabling "analyst upgrade momentum"
  roughly doubles the API calls needed and will slow down scans.
- The S&P 500 constituent list is scraped from Wikipedia and cached for 24 hours.
- Analyst estimates (price targets, growth estimates, ratings) reflect current
  consensus and change frequently.
- This tool has **no view on macro risk, earnings surprises, litigation,
  management changes**, or other qualitative factors that often drive the
  biggest stock moves.
