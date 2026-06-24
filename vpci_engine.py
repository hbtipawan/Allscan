#!/usr/bin/env python3
"""
════════════════════════════════════════════════════════════════════════════════
VPCI v3 — Shared Analysis Engine
════════════════════════════════════════════════════════════════════════════════
Common functions used by all three screeners (NSE, US, ETF).
Place this file in the same folder as the screener scripts.
════════════════════════════════════════════════════════════════════════════════
"""

import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# Minimum weekly bars needed (40w SMA needs 40, VPCI long needs 20, ATR needs 14)
# 42 bars captures ~10 months — enough for all indicators except 52w high (gracefully handled)
MIN_BARS = 42


# ═══════════════════════════════════════════════════════════════════════════════
# BULLETPROOF DATA FETCHING
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_weekly_data(yf_symbol, av_symbol=None, av_key="demo"):
    """
    Multi-Tier weekly data fetch. Never silently skips.
      Tier 1: yfinance (yf_symbol)
      Tier 2: Alpha Vantage fallback (av_symbol)
    Normalizes columns, filters incomplete week, drops bad rows.
    """
    df = pd.DataFrame()

    # Tier 1: Yahoo Finance
    try:
        stock = yf.Ticker(yf_symbol)
        df = stock.history(period="2y", interval="1wk")
    except Exception:
        pass

    # Tier 2: Alpha Vantage
    if (df is None or df.empty or len(df) < MIN_BARS) and av_symbol and av_key:
        try:
            import requests
            url = (f"https://www.alphavantage.co/query?"
                   f"function=TIME_SERIES_WEEKLY&symbol={av_symbol}&apikey={av_key}")
            res = requests.get(url, timeout=10)
            data = res.json()
            if "Weekly Time Series" in data:
                av_df = pd.DataFrame.from_dict(data["Weekly Time Series"], orient="index")
                av_df = av_df.rename(columns={
                    "1. open": "Open", "2. high": "High",
                    "3. low": "Low", "4. close": "Close", "5. volume": "Volume"
                })
                av_df.index = pd.to_datetime(av_df.index)
                av_df = av_df.astype(float).sort_index()
                df = av_df
        except Exception:
            pass

    if df is None or df.empty or len(df) < MIN_BARS:
        return None

    # Normalize columns
    df.columns = [str(c).strip().capitalize() for c in df.columns]
    for col in ["Close", "High", "Low", "Volume", "Open"]:
        if col not in df.columns:
            return None

    # Drop incomplete current week
    if len(df) > 2:
        last_vol = df["Volume"].iloc[-1]
        prev_vol = df["Volume"].iloc[-2]
        if prev_vol > 0 and last_vol < (prev_vol * 0.25):
            df = df.iloc[:-1]

    df = df[(df["Close"] > 0) & (df["Volume"] > 0)].dropna(subset=["Close", "Volume"])
    return df if len(df) >= MIN_BARS else None


# ═══════════════════════════════════════════════════════════════════════════════
# VPCI v3 ANALYSIS + LATEST CANDLE SIGNAL DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def _check_gates_at(close_pr, high_pr, ef, es, s40, vpci_val, vpci_sig, ret_rs, hi_brk, params, idx=-1):
    """Check all 7 gates at a specific bar index. Returns (gates_list, count)."""
    c       = float(close_pr.iloc[idx])
    l_ef    = float(ef.iloc[idx])
    l_es    = float(es.iloc[idx])
    l_s40   = float(s40.iloc[idx]) if pd.notna(s40.iloc[idx]) else 0
    l_vpci  = float(vpci_val.iloc[idx]) if pd.notna(vpci_val.iloc[idx]) else 0
    p_vpci  = float(vpci_val.iloc[idx - 1]) if pd.notna(vpci_val.iloc[idx - 1]) else 0
    l_vsig  = float(vpci_sig.iloc[idx]) if pd.notna(vpci_sig.iloc[idx]) else 0
    l_hi_brk= float(hi_brk.iloc[idx]) if pd.notna(hi_brk.iloc[idx]) else c
    l_ret   = float(ret_rs.iloc[idx]) if pd.notna(ret_rs.iloc[idx]) else 0
    slope_w = params["slope_weeks"]
    ef_slope= l_ef > float(ef.iloc[idx - slope_w]) if abs(idx) + slope_w <= len(ef) else False

    g1 = bool(l_ef > l_es)
    g2 = bool(c > l_ef)
    g3 = bool(ef_slope)
    g4 = bool(c > l_hi_brk)
    g5 = bool(l_vpci > 0 and l_vpci > l_vsig and l_vpci > p_vpci)
    g6 = bool(l_ret > 0)
    g7 = bool(c > l_s40 and l_ef > l_s40) if l_s40 > 0 else False

    return [g1, g2, g3, g4, g5, g6, g7], sum([g1, g2, g3, g4, g5, g6, g7])


def analyze_stock_v3(symbol, df, params):
    """Full v3 7-gate analysis with latest-candle signal detection."""
    try:
        close_pr = df["Close"]
        high_pr  = df["High"]
        low_pr   = df["Low"]
        vol      = df["Volume"]
        open_pr  = df["Open"]

        ef  = close_pr.ewm(span=params["ema_fast"], adjust=False).mean()
        es  = close_pr.ewm(span=params["ema_slow"], adjust=False).mean()
        s40 = close_pr.rolling(window=params["sma_long"]).mean()

        tr = pd.concat([high_pr - low_pr,
                        abs(high_pr - close_pr.shift(1)),
                        abs(low_pr - close_pr.shift(1))], axis=1).max(axis=1)
        atr_s = tr.rolling(window=params["atr_len"]).mean()

        vpciS, vpciL = params["vpci_short"], params["vpci_long"]
        cv = close_pr * vol
        vwma_L   = cv.rolling(vpciL).sum() / vol.rolling(vpciL).sum()
        vwma_S   = cv.rolling(vpciS).sum() / vol.rolling(vpciS).sum()
        sma_C_L  = close_pr.rolling(vpciL).mean()
        sma_C_S  = close_pr.rolling(vpciS).mean()
        sma_V_L  = vol.rolling(vpciL).mean()
        sma_V_S  = vol.rolling(vpciS).mean()
        vpc      = vwma_L - sma_C_L
        vpr      = vwma_S / sma_C_S
        vm       = sma_V_S / sma_V_L
        vpci_val = vpc * vpr * vm
        vpci_vol = vpci_val * vol
        vpci_sig = vpci_vol.rolling(vpciS).sum() / vol.rolling(vpciS).sum()

        hi_brk    = high_pr.rolling(params["brk_weeks"]).max().shift(1)
        ret_rs    = close_pr.pct_change(periods=params["rs_weeks"])
        # 52w high — use min_periods=MIN_BARS so stocks with <52 weeks still get a value
        hi52w     = high_pr.rolling(52, min_periods=MIN_BARS).max()
        swing_low = low_pr.rolling(params["swing_lookback"]).min()
        candle_rng= high_pr - low_pr

        # Safe value extraction
        def safe(s, idx=-1):
            v = s.iloc[idx]
            return float(v) if pd.notna(v) else None

        c       = safe(close_pr)
        o       = safe(open_pr)
        l_ef    = safe(ef)
        l_es    = safe(es)
        l_s40   = safe(s40) or 0
        l_atr   = safe(atr_s) or (c * 0.05)
        l_vpci  = safe(vpci_val) or 0
        p_vpci  = safe(vpci_val, -2) or 0
        l_vsig  = safe(vpci_sig) or 0
        l_hi_brk= safe(hi_brk) or c
        l_ret   = safe(ret_rs) or 0
        l_hi52w = safe(hi52w) or c
        l_swing = safe(swing_low, -2) or (c * 0.9)
        l_candle= safe(candle_rng)

        pct_from_52w = (l_hi52w - c) / l_hi52w * 100 if l_hi52w > 0 else 0

        # ─── Current bar gates ───
        gates, gate_count = _check_gates_at(
            close_pr, high_pr, ef, es, s40, vpci_val, vpci_sig, ret_rs, hi_brk, params, idx=-1)
        g1, g2, g3, g4, g5, g6, g7 = gates
        tier1_pass = g4 and g5 and g7

        ext_f        = params["ext_filter"]
        not_extended = True if ext_f == 0 else l_candle <= ext_f * l_atr
        is_extended  = (not not_extended) and (ext_f > 0)
        limit_entry  = float(close_pr.iloc[-2]) + 0.5 * l_atr if len(close_pr) > 1 else c

        full_entry      = all(gates) and not_extended
        relaxed_entry   = params["relaxed"] and gate_count >= 6 and tier1_pass and not_extended and not full_entry
        is_entry        = full_entry or relaxed_entry
        gates_ready_ext = all(gates) and is_extended

        # ─── LATEST CANDLE SIGNAL DETECTION ───
        # Check if the PREVIOUS bar did NOT have an entry signal but current bar DOES
        # This means the BUY signal just fired on the latest (most recent) candlestick
        fresh_signal = False
        if is_entry and len(close_pr) > 2:
            try:
                prev_gates, prev_gc = _check_gates_at(
                    close_pr, high_pr, ef, es, s40, vpci_val, vpci_sig, ret_rs, hi_brk, params, idx=-2)
                # Previous bar candle range for extended check
                prev_candle = float(candle_rng.iloc[-2]) if pd.notna(candle_rng.iloc[-2]) else 0
                prev_atr    = float(atr_s.iloc[-2]) if pd.notna(atr_s.iloc[-2]) else l_atr
                prev_not_ext= True if ext_f == 0 else prev_candle <= ext_f * prev_atr
                prev_full   = all(prev_gates) and prev_not_ext
                prev_relaxed= params["relaxed"] and prev_gc >= 6 and (prev_gates[3] and prev_gates[4] and prev_gates[6]) and prev_not_ext and not prev_full
                prev_entry  = prev_full or prev_relaxed
                fresh_signal = not prev_entry  # signal is FRESH if previous bar had no entry
            except Exception:
                fresh_signal = False

        # Also detect fresh extended signal (7/7 but extended just appeared)
        fresh_ext_signal = False
        if gates_ready_ext and len(close_pr) > 2:
            try:
                prev_gates, _ = _check_gates_at(
                    close_pr, high_pr, ef, es, s40, vpci_val, vpci_sig, ret_rs, hi_brk, params, idx=-2)
                prev_all_pass = all(prev_gates)
                fresh_ext_signal = not prev_all_pass
            except Exception:
                fresh_ext_signal = False

        # ─── Risk levels ───
        trail_stop = c - params["trail_mult"] * l_atr
        init_sl    = c - params["atr_sl_mult"] * l_atr
        risk_pct   = (c - trail_stop) / c * 100 if c > 0 else 0

        avg_vol   = float(sma_V_L.iloc[-1]) if pd.notna(sma_V_L.iloc[-1]) else 1
        vol_ratio = float(vol.iloc[-1]) / avg_vol if avg_vol > 0 else 0
        is_up     = c > o

        if vol_ratio > 1 and is_up:       candle_type = "Strong Demand"
        elif vol_ratio > 1 and not is_up:  candle_type = "Strong Supply"
        elif vol_ratio <= 1 and is_up:     candle_type = "Waning Demand"
        else:                              candle_type = "Drying Supply"

        price_up = c > float(close_pr.iloc[-2]) if len(close_pr) > 1 else False
        vpci_up  = l_vpci > p_vpci
        if price_up and vpci_up:           scenario = "Bull Confirm"
        elif price_up and not vpci_up:     scenario = "Bear Contradict"
        elif not price_up and vpci_up:     scenario = "Bull Contradict"
        else:                              scenario = "Bear Confirm"

        gate_labels = ["G1:EMA×", "G2:>EMA", "G3:Slope", "G4:Brkout", "G5:VPCI", "G6:RS", "G7:>40w"]
        missed = [gate_labels[i] for i in range(7) if not gates[i]]

        return {
            "symbol": symbol, "close": round(c, 2), "gate_count": gate_count,
            "g1": g1, "g2": g2, "g3": g3, "g4": g4, "g5": g5, "g6": g6, "g7": g7,
            "tier1_pass": tier1_pass, "full_entry": full_entry,
            "relaxed_entry": relaxed_entry, "is_entry": is_entry,
            "not_extended": not_extended, "is_extended": is_extended,
            "gates_ready_ext": gates_ready_ext, "limit_entry": round(limit_entry, 2),
            "fresh_signal": fresh_signal, "fresh_ext_signal": fresh_ext_signal,
            "pct_from_52w": round(pct_from_52w, 1),
            "pct_near_52w": round(100 - pct_from_52w, 1),
            "vpci": round(l_vpci, 4), "rs_return": round(l_ret * 100, 1),
            "ema_fast": round(l_ef, 2), "ema_slow": round(l_es, 2),
            "sma40": round(l_s40, 2), "atr": round(l_atr, 2),
            "trail_stop": round(trail_stop, 2), "init_sl": round(init_sl, 2),
            "swing_low": round(l_swing, 2), "risk_pct": round(risk_pct, 1),
            "vol_ratio": round(vol_ratio, 2), "candle_type": candle_type,
            "scenario": scenario, "missed_gates": ", ".join(missed),
            "avg_volume": int(avg_vol),
        }
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# DISPLAY + AUTO-SAVE
# ═══════════════════════════════════════════════════════════════════════════════

def print_and_save_results(results, top_n=30, relaxed=False, market="NSE", currency="Rs."):
    """Display results AND auto-save full CSV."""
    if not results:
        print("\n  No results to display or save.")
        return None

    df = pd.DataFrame(results)
    total = len(df)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    # ─── 🔥 FRESH BUY SIGNALS (latest candle) ───
    fresh = df[df["fresh_signal"]].sort_values("pct_near_52w", ascending=False)
    if len(fresh) > 0:
        print(f"\n{'═' * 90}")
        print(f"  🔥 FRESH BUY SIGNAL on LATEST CANDLE: {len(fresh)}")
        print(f"     (Signal just fired this week — was NOT active last week)")
        print(f"{'═' * 90}")
        for _, r in fresh.head(top_n).iterrows():
            etype = "7/7" if r["full_entry"] else "6/7"
            print(f"  {r['symbol']:<15s} | {currency}{r['close']:<10.2f} | [{etype}] "
                  f"| VPCI: {r['vpci']:>7.4f} | RS: {r['rs_return']:>+6.1f}% "
                  f"| 52wH: {r['pct_from_52w']:>5.1f}% away "
                  f"| Trail: {currency}{r['trail_stop']:<9.2f} | Risk: {r['risk_pct']:.1f}% | {r['scenario']}")

    # ─── 🔥 FRESH EXTENDED SIGNALS ───
    fresh_ext = df[df["fresh_ext_signal"]].sort_values("pct_near_52w", ascending=False)
    if len(fresh_ext) > 0:
        print(f"\n{'═' * 90}")
        print(f"  ⚡ FRESH 7/7 BUT EXTENDED (just appeared this week): {len(fresh_ext)}")
        print(f"{'═' * 90}")
        for _, r in fresh_ext.head(top_n).iterrows():
            print(f"  {r['symbol']:<15s} | {currency}{r['close']:<10.2f} "
                  f"| VPCI: {r['vpci']:>7.4f} | RS: {r['rs_return']:>+6.1f}% "
                  f"| LIMIT: {currency}{r['limit_entry']:<9.2f} | {r['scenario']}")

    # ─── ★ BUYABLE 7/7 ───
    buyable = df[df["full_entry"]].sort_values("pct_near_52w", ascending=False)
    print(f"\n{'═' * 90}")
    print(f"  ★ BUYABLE (7/7 Gates Passed): {len(buyable)}")
    print(f"{'═' * 90}")
    if len(buyable) > 0:
        for _, r in buyable.head(top_n).iterrows():
            fresh_tag = " 🔥NEW" if r.get("fresh_signal") else ""
            print(f"  {r['symbol']:<15s} | {currency}{r['close']:<10.2f} | VPCI: {r['vpci']:>7.4f} "
                  f"| RS: {r['rs_return']:>+6.1f}% | 52wH: {r['pct_from_52w']:>5.1f}% away "
                  f"| Trail: {currency}{r['trail_stop']:<9.2f} | Risk: {r['risk_pct']:.1f}% "
                  f"| {r['scenario']}{fresh_tag}")
    else:
        print("  No stocks passed all 7 gates today.")

    # ─── ⚡ EXTENDED BUT READY ───
    ext_ready = df[df["gates_ready_ext"]].sort_values("pct_near_52w", ascending=False)
    if len(ext_ready) > 0:
        print(f"\n{'═' * 90}")
        print(f"  ⚡ 7/7 GATES BUT EXTENDED — Place LIMIT BUY: {len(ext_ready)}")
        print(f"{'═' * 90}")
        for _, r in ext_ready.head(top_n).iterrows():
            fresh_tag = " 🔥NEW" if r.get("fresh_ext_signal") else ""
            print(f"  {r['symbol']:<15s} | {currency}{r['close']:<10.2f} | VPCI: {r['vpci']:>7.4f} "
                  f"| RS: {r['rs_return']:>+6.1f}% | LIMIT: {currency}{r['limit_entry']:<9.2f} "
                  f"| {r['scenario']}{fresh_tag}")

    # ─── RELAXED 6/7 ───
    if relaxed:
        rel = df[df["relaxed_entry"]].sort_values("pct_near_52w", ascending=False)
        if len(rel) > 0:
            print(f"\n{'═' * 90}")
            print(f"  ★ RELAXED ENTRY (6/7, Tier 1 Passed): {len(rel)}")
            print(f"{'═' * 90}")
            for _, r in rel.head(top_n).iterrows():
                fresh_tag = " 🔥NEW" if r.get("fresh_signal") else ""
                print(f"  {r['symbol']:<15s} | {currency}{r['close']:<10.2f} | VPCI: {r['vpci']:>7.4f} "
                      f"| Missed: {r['missed_gates']}{fresh_tag}")

    # ─── ◉ WATCHLIST 6/7 ───
    watch = df[(df["gate_count"] >= 6) & (~df["is_entry"]) & df["tier1_pass"]].sort_values("pct_near_52w", ascending=False)
    print(f"\n{'═' * 90}")
    print(f"  ◉ WATCHLIST (6/7 Gates, Tier 1 Passed): {len(watch)}")
    print(f"{'═' * 90}")
    if len(watch) > 0:
        for _, r in watch.head(top_n).iterrows():
            print(f"  {r['symbol']:<15s} | {currency}{r['close']:<10.2f} | VPCI: {r['vpci']:>7.4f} "
                  f"| RS: {r['rs_return']:>+6.1f}% | Missed: {r['missed_gates']} | {r['scenario']}")
    else:
        print("  None.")

    # ─── ▲ STRONG MOMENTUM 5+ ───
    strong = df[df["gate_count"] >= 5].sort_values(["gate_count", "pct_near_52w"], ascending=[False, False])
    print(f"\n{'═' * 90}")
    print(f"  ▲ STRONG MOMENTUM (5+ Gates): {len(strong)} total, showing top {top_n}")
    print(f"{'═' * 90}")
    if len(strong) > 0:
        for _, r in strong.head(top_n).iterrows():
            print(f"  {r['symbol']:<15s} | {currency}{r['close']:<10.2f} | Gates: {r['gate_count']}/7 "
                  f"| VPCI: {r['vpci']:>7.4f} | RS: {r['rs_return']:>+6.1f}% "
                  f"| 52wH: {r['pct_from_52w']:>5.1f}% away | {r['scenario']}")

    # ─── BREADTH ───
    print(f"\n{'═' * 90}")
    print(f"  {market} MARKET BREADTH")
    print(f"{'═' * 90}")
    print(f"  Total screened:     {total}")
    for label, cnt in [
        ("7/7 gates", len(df[df["gate_count"] == 7])),
        ("6/7 gates", len(df[df["gate_count"] == 6])),
        ("5+ gates", len(df[df["gate_count"] >= 5])),
        ("Above 40w (G7)", len(df[df["g7"]])),
        ("VPCI > 0", len(df[df["vpci"] > 0])),
        ("Within 10% 52wH", len(df[df["pct_from_52w"] <= 10])),
        ("Fresh BUY signals", len(df[df["fresh_signal"]])),
    ]:
        print(f"  {label:<18s}: {cnt:>4d} ({cnt/total*100:>5.1f}%)")

    print(f"\n  Scenario Distribution:")
    for s in ["Bull Confirm", "Bull Contradict", "Bear Contradict", "Bear Confirm"]:
        cnt = len(df[df["scenario"] == s])
        bar = "█" * int(cnt / total * 40) if total > 0 else ""
        print(f"    {s:18s}: {cnt:>4d} ({cnt/total*100:>5.1f}%) {bar}")
    print(f"{'═' * 90}")

    # ─── AUTO-SAVE ALL RESULTS TO CSV ───
    df_sorted = df.sort_values(["gate_count", "pct_near_52w"], ascending=[False, False])

    def status_label(row):
        if row.get("fresh_signal"):      return "🔥 FRESH BUY"
        if row.get("fresh_ext_signal"):  return "🔥 FRESH EXT"
        if row["full_entry"]:            return "★ BUYABLE (7/7)"
        if row["gates_ready_ext"]:       return "⚡ 7/7 EXTENDED"
        if row.get("relaxed_entry"):     return "★ RELAXED (6/7)"
        if row["gate_count"] >= 6 and row["tier1_pass"]: return "◉ WATCHLIST (6/7)"
        if row["gate_count"] >= 5:       return "▲ MOMENTUM (5+)"
        return ""

    df_sorted["status"] = df_sorted.apply(status_label, axis=1)

    priority_cols = [
        "status", "symbol", "close", "gate_count", "fresh_signal", "fresh_ext_signal",
        "g1", "g2", "g3", "g4", "g5", "g6", "g7",
        "pct_from_52w", "pct_near_52w", "vpci", "rs_return",
        "trail_stop", "init_sl", "swing_low", "risk_pct", "limit_entry",
        "vol_ratio", "candle_type", "scenario", "missed_gates",
        "not_extended", "is_extended", "gates_ready_ext",
        "full_entry", "relaxed_entry", "is_entry", "tier1_pass",
        "ema_fast", "ema_slow", "sma40", "atr", "avg_volume",
    ]
    existing = [c for c in priority_cols if c in df_sorted.columns]
    remaining = [c for c in df_sorted.columns if c not in existing]
    df_sorted = df_sorted[existing + remaining]

    csv_name = f"vpci_{market.lower()}_results_{timestamp}.csv"
    csv_latest = f"vpci_{market.lower()}_results_latest.csv"
    df_sorted.to_csv(csv_name, index=False)
    df_sorted.to_csv(csv_latest, index=False)
    print(f"\n  [✓] ALL {len(df_sorted)} results saved to: {csv_name}")
    print(f"  [✓] Latest copy: {csv_latest}\n")

    return df_sorted


# ═══════════════════════════════════════════════════════════════════════════════
# DEFAULT PARAMS
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_PARAMS = {
    "ema_fast": 13, "ema_slow": 26, "sma_long": 40,
    "slope_weeks": 3, "brk_weeks": 13, "rs_weeks": 13,
    "vpci_short": 5, "vpci_long": 20, "atr_len": 14,
    "trail_mult": 3.0, "atr_sl_mult": 1.5,
    "ext_filter": 3.0, "swing_lookback": 8,
    "relaxed": False, "av_key": "demo",
}
