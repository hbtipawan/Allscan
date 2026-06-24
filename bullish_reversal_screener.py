#!/usr/bin/env python3
"""
bullish_reversal_screener.py — command-line runner (no Streamlit needed).
Uses the SAME engine as the app (screener_core.py), so results are identical.

Examples:
  python3 bullish_reversal_screener.py --selftest          # run detector unit tests
  python3 bullish_reversal_screener.py --max 200           # scan first 200 NSE names
  python3 bullish_reversal_screener.py --bse --max 150     # scan BSE
  python3 bullish_reversal_screener.py --both --scan 2     # NSE+BSE, last 2 bars
Data source: Yahoo by default. To use Dhan, set DHAN_ACCESS_TOKEN (and optionally
DHAN_CLIENT_ID) in your environment — the runner then maps symbols to Dhan security
IDs automatically.
Writes bullish_screener_report.html.
"""
import sys, os, argparse, pandas as pd
import screener_core as sc

NSE_CSV, BSE_CSV = "EQUITY_L_2.csv", "bse_stocks.csv"

def load_nse(path=NSE_CSV):
    d = pd.read_csv(path).rename(columns=lambda c: str(c).strip())
    out = pd.DataFrame({"symbol": d["companyId"].astype(str).str.strip(),
                        "name": d["Name"].astype(str).str.strip(),
                        "sector": d.get("Sector", "").astype(str).str.strip(), "exch": "NSE"})
    out["yahoo"] = out["symbol"] + ".NS"
    return out

def load_bse(path=BSE_CSV):
    d = pd.read_csv(path).rename(columns=lambda c: str(c).strip())
    if "Status" in d: d = d[d["Status"].astype(str).str.strip().str.lower() == "active"]
    out = pd.DataFrame({"symbol": d["Scrip ID"].astype(str).str.strip(),
                        "name": d["Scrip Name"].astype(str).str.strip(),
                        "sector": d.get("Industry", "").astype(str).str.strip(), "exch": "BSE"})
    out["yahoo"] = out["symbol"] + ".BO"
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--bse", action="store_true")
    ap.add_argument("--both", action="store_true")
    ap.add_argument("--max", type=int, default=150)
    ap.add_argument("--all", action="store_true", help="scan every matching stock (no cap)")
    ap.add_argument("--scan", type=int, default=1)
    ap.add_argument("--weekly", action="store_true", help="use weekly candles instead of daily")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    if a.selftest:
        sys.exit(0 if sc.selftest() else 1)

    frames = []
    if a.both or not a.bse: frames.append(load_nse())
    if a.both or a.bse:     frames.append(load_bse())
    uni = pd.concat(frames, ignore_index=True).drop_duplicates("yahoo")
    if not a.all:
        uni = uni.head(a.max)
    rows = uni.to_dict("records")

    interval = "1wk" if a.weekly else "1d"
    rng = "5y" if a.weekly else "1y"
    tf = "Weekly" if a.weekly else "Daily"
    weekly = a.weekly

    token = os.environ.get("DHAN_ACCESS_TOKEN", "").strip()
    client = os.environ.get("DHAN_CLIENT_ID", "").strip() or None
    if token:
        import dhan_data as dd
        print("Mapping symbols to Dhan security IDs...", file=sys.stderr)
        nse_map, bse_map = dd.build_symbol_maps(dd.load_scrip_master())
        uni, unmapped = dd.map_universe(uni, nse_map, bse_map)
        rows = uni.to_dict("records")
        if unmapped:
            print(f"  {len(unmapped)} symbols not on Dhan, skipped.", file=sys.stderr)
        years = 2.5 if weekly else 1.2
        fetch_fn = dd.make_fetch_fn(token, weekly=weekly, years=years, client_id=client)
        print(f"Data source: Dhan ({len(rows)} mapped symbols)", file=sys.stderr)
    else:
        fetch_fn = lambda row: sc.fetch_ohlcv(row["yahoo"], rng=rng, interval=interval)
        print("Data source: Yahoo (set DHAN_ACCESS_TOKEN to use Dhan)", file=sys.stderr)
    eff_workers, req_delay = (min(a.workers, 4 if token else 6), 0.2 if token else 0.15) \
                              if len(rows) > 500 else (a.workers, 0.05 if token else 0.0)

    done = {"n": 0}
    def prog(d, t, s):
        done["n"] = d
        if d % 20 == 0 or d == t: print(f"  {d}/{t}", file=sys.stderr)
    print(f"Screening {len(rows)} symbols ({tf} candles)...", file=sys.stderr)
    results, scanned, failed = sc.run_screen(rows, fetch_fn=fetch_fn, scan_last_n=a.scan,
                                             max_workers=eff_workers, progress=prog,
                                             request_delay=req_delay)
    html = sc.build_html(results, scanned, failed, a.scan, timeframe=tf)
    open("bullish_screener_report.html", "w").write(html)
    total = sum(len(v) for v in results.values())
    print(f"\nDone. {scanned} scanned, {total} signals -> bullish_screener_report.html")
    for name, wp, *_ in sc.PATTERNS:
        if results[name]:
            print(f"  {wp:>3}%  {name:<26} {len(results[name])} -> "
                  + ", ".join(r['symbol'] for r in results[name][:8]))

if __name__ == "__main__":
    main()
