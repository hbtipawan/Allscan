# Bullish Scanners (NSE / BSE)

Three scanners in one Streamlit app:

1. **Reversal patterns** — 16 bullish reversal candle/volume patterns detected at
   *lows* (after a downtrend), tabs ordered by historical reversal frequency.
2. **VCP breakout** — high-grade Minervini-style volatility-contraction bases near
   the *highs*: tight, contracting, quiet bases that are **Coiling** (ready under the
   pivot) or **Breaking out** (clearing the pivot today on a volume surge).
3. **VPCI weekly (7-gate)** — the weekly trend + volume-price-confirmation engine.
   A 7/7 stock passes EMA structure, price-above-EMA, EMA slope, 13-week breakout,
   VPCI accumulation, positive relative strength, and price above the 40-week MA.
   Results group into **Fresh / Buyable (7/7) / Watchlist (6/7) / All / Ranked /
   G4-pending**, plus **Sector Leadership** and **Sector Rotation** tabs. Always
   weekly; the original scanner's earnings-phase tagging and US/ETF markets are not
   included in this build. Its scoring/gate logic is unchanged from `vpci_engine.py`.

Pick the scanner with the **Mode** toggle at the top of the sidebar. All three share the
Exchange, data source, full-universe option, sector filter, and scan controls.

## Repo layout (put all of these in the GitHub repo root)

```
streamlit_app.py              # the Streamlit UI (both scanners, Upstox/Dhan/Yahoo)
screener_core.py              # reversal-pattern engine
vcp_core.py                   # VCP engine (backtest-tuned)
upstox_data.py                # Upstox data layer (free, no token) — recommended
dhan_data.py                  # Dhan data layer (paid Data API + token/TOTP)
marketcap.py                  # market-cap lookup (₹ Crore) for result stocks
vpci_engine.py                # VPCI v3 weekly 7-gate engine (UNCHANGED core)
sector_history.py             # sector leadership / rotation / snapshots (UNCHANGED)
vpci_scanner.py               # wires VPCI into the app: Upstox feed + tabs (no earnings/US)
bullish_reversal_screener.py  # optional command-line reversal runner
requirements.txt
EQUITY_L_2.csv                # NSE list  (cols: companyId, Name, Sector, Industry)
bse_stocks.csv                # BSE list  (cols: Scrip Code, Scrip ID, Scrip Name, Status, ...)

# analysis (not needed to run the app):
vcp_backtest.py               # the breakout backtest harness
VCP_BACKTEST_FINDINGS.md      # results + recommended settings
```

> The two CSV filenames are referenced at the top of `streamlit_app.py`
> (`NSE_CSV`, `BSE_CSV`). If you rename the files, update those two lines.

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy on Streamlit Community Cloud (free)

1. Push the repo (all files above) to GitHub.
2. Go to https://share.streamlit.io → **New app** → pick your repo/branch.
3. Set **Main file path** to `streamlit_app.py`. Deploy.

That's it — `requirements.txt` is picked up automatically.

## Data source: Upstox (recommended), Dhan, or Yahoo

Pick the source with the **Data source** toggle in the sidebar.

**Upstox (default, recommended).** Upstox's historical-candle API is **free, needs no token,
no login, and no subscription**, and it's exchange-grade (adjusted for splits/bonuses). The
app downloads Upstox's public instrument list once (cached for a day) and maps your NSE/BSE
symbols automatically. Nothing to configure — just run.

**Dhan.** Accurate, but historical data needs a paid **Data API subscription** (₹499 + tax/
month, or free if you've done 25+ trades in the last 30 days) **and** a 24-hour access token.
Use the **"Check Dhan connection"** button to see whether your token is valid and your Data
API plan is active. Two login modes: *Paste token* (24h JWT, or set `DHAN_ACCESS_TOKEN` in
secrets), or *Auto-login (TOTP)* — set `DHAN_CLIENT_ID`, `DHAN_PIN`, `DHAN_TOTP_SECRET` in
secrets (TOTP must be enabled on your Dhan account) and the app mints a fresh token daily.

**Yahoo.** Free, no token, but less accurate for Indian stocks (misses some corporate-action
adjustments).

## CLI data source

The command-line runner uses Yahoo by default. To use Dhan, export your token first:
```bash
export DHAN_ACCESS_TOKEN="your_jwt_here"
export DHAN_CLIENT_ID="your_client_id"   # optional
python3 bullish_reversal_screener.py --all
```

## Using it

- **Exchange**: NSE, BSE, or Both (Both prefers NSE for dual-listed names).
- **Timeframe**: Daily or Weekly candles. Weekly applies the same patterns and
  downtrend gate to weekly bars (fetches ~5y of history) — fewer, slower, but more
  significant signals. The latest weekly candle is the in-progress week.
- **Sectors**: optional filter (NSE Sector / BSE Industry).
- **Scan entire NSE/BSE universe (ignore my list)**: an add-on toggle. When on, the app
  screens *every* cash-equity on the chosen exchange(s) pulled straight from the data
  source (≈3,200 NSE, ≈6,000 BSE) instead of your uploaded CSVs — useful when you don't
  want to maintain a list. Your CSVs are still the default when it's off. (Sector filtering
  isn't available in full-universe mode, since the master lists carry no sector field.)
- **Max stocks to scan**: cap for a quick run, or tick **Scan ALL matching stocks**
  to scan the entire filtered list in one pass. Large scans (>500) auto-throttle
  (fewer threads + a small per-request delay) to respect data-source rate limits
  (stricter on Dhan) — they take a few minutes the first time, then cache for the
  day so re-runs are instant.
- **Scan last N bars**: 1 = only the freshest completed bar (default). 2–3 catches
  signals from the last few sessions.
- **Volume-confirmed only**: show only signals whose signal-bar volume >=1.5x its
  20-day average.
- Every result row shows a **Market Cap (₹ Crore)** column. It's fetched only for the
  result stocks (not the whole universe), so it's quick; it shows a dash if unavailable.
- **Run scan** -> results appear. Each **stock symbol is a clickable link that opens
  its TradingView chart** in a new tab. There's also a button to download a full
  standalone HTML report (symbols clickable there too).

## VCP scanner settings

- **Within % of 52-period high** (default 25%, Minervini standard) — only stocks near
  their highs qualify.
- **Max base tightness %** (default 5%) — the final contraction must be no wider.
- **Min base length** — minimum bars in the tight base.
- **Trend strictness** — *Strict* = textbook Stage 2 (price > 50 > 150 > 200 MA, 200-MA
  rising); *Standard*/*Relaxed* loosen it for recovery setups.
- **Minimum grade** (A/B/C) and **Status** (Coiling/Breakout) filters.

The VCP scan also gates on **volatility contraction** (recent range narrower than
earlier) and **volume dry-up** (base quieter than its 50-bar average), and grades each
base A/B/C from tightness, contraction, dry-up, proximity to high, base length, and
relative strength vs Nifty. It is intentionally strict — on any given day only a
handful of stocks are in a genuine high-grade VCP. Stocks well below their 52-week high
(recovery breakouts) are excluded by design.

## Notes on accuracy

- Symbol mapping: NSE → `SYMBOL.NS`; BSE → `SCRIP_ID.BO` (the alphabetic Scrip ID,
  not the numeric code — that's what Yahoo serves reliably).
- Detection logic lives only in `screener_core.py` and is covered by synthetic
  unit tests: `python3 screener_core.py` (or `--selftest` on the CLI) checks that
  every detector fires after a downtrend and is *rejected* in an uptrend.
- The win-% on each tab is a **historical best-case reversal frequency** (a ranking
  aid), not a tradeable win-rate after costs. This is a research/screening tool,
  not investment advice.
```
