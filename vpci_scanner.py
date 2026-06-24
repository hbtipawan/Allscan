#!/usr/bin/env python3
"""
vpci_scanner.py — integrates the VPCI v3 weekly 7-gate scanner into the multi-scanner app.

Core scoring/gate logic is UNCHANGED: it calls vpci_engine.analyze_stock_v3 and reuses the
exact ranker (rank_stocks / rank_g4_pending) from the original app. This module only:
  * feeds the engine WEEKLY OHLCV from the app's data layer (Upstox/Dhan/Yahoo),
  * fills the Market Cap column from the shared marketcap helper (Screener, cloud-safe),
  * renders the result tabs (Fresh / Buyable / Watchlist / All / Ranked / G4 / Sector
    Leadership / Sector Rotation) — the Phase (earnings) tab and US markets are removed.
"""
import re
from datetime import datetime, date as _date
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import streamlit as st

from vpci_engine import analyze_stock_v3, DEFAULT_PARAMS
from sector_history import (
    load_sector_map, attach_sector_columns, build_sector_leadership,
    save_weekly_snapshot, load_history_from_github, build_rotation_view,
)

# ════════════════════ scoring helpers (verbatim from original app) ════════════════════
def parse_mcap_to_crore(mcap_str):
    if pd.isna(mcap_str) or mcap_str == "N/A" or not str(mcap_str).strip():
        return np.nan
    s = str(mcap_str).lower().replace("crore inr", "").replace("inr", "").strip()
    m = re.match(r"([\d.]+)\s*k", s)
    if m: return float(m.group(1)) * 1000
    m = re.match(r"([\d.]+)", s)
    if m: return float(m.group(1))
    return np.nan

def mcap_score(crore):
    if pd.isna(crore) or crore <= 0: return 0.5
    if 5000 <= crore <= 15000: return 1.0
    if 2000 <= crore < 5000: return 0.6 + 0.4 * (crore - 2000) / 3000
    if 15000 < crore <= 30000: return 1.0 - 0.4 * (crore - 15000) / 15000
    if 1000 <= crore < 2000: return 0.3 + 0.3 * (crore - 1000) / 1000
    if 30000 < crore <= 100000: return 0.6 - 0.6 * (crore - 30000) / 70000
    return 0.0

def phase_score(phase_tag):
    return {"POST_SWEET":1.00,"POST_HOT":0.75,"PRE_HOT":0.60,"STANDARD":0.35,
            "PRE_IMMINENT":0.55,"POST_FADING":0.25,"PRE_AVOID":0.20,"UNKNOWN":0.35}.get(phase_tag, 0.35)

def rank_stocks(df, include_relaxed=False, min_gates=7):
    if include_relaxed:
        mask = (df["full_entry"] == True) | (df["relaxed_entry"] == True) | (df["gate_count"] >= min_gates)
    else:
        mask = df["full_entry"] == True
    pool = df[mask].copy().reset_index(drop=True)
    if len(pool) == 0: return pool
    pool["score_vpci"]  = pool["vpci"].rank(pct=True)
    pool["score_rs"]    = pool["rs_return"].rank(pct=True)
    pool["score_52w"]   = (pool["pct_near_52w"].clip(0, 100) / 100.0)
    pool["score_tight"] = 1.0 - pool["risk_pct"].rank(pct=True)
    pool["score_vol"]   = pool["vol_ratio"].rank(pct=True) if "vol_ratio" in pool.columns else 0.5
    pool["mcap_crore"]  = pool["Market Cap"].apply(parse_mcap_to_crore)
    pool["score_mcap"]  = pool["mcap_crore"].apply(mcap_score)
    pool["score_phase"] = pool["phase"].apply(phase_score) if "phase" in pool.columns else 0.35
    weights = {"score_vpci":0.22,"score_rs":0.22,"score_52w":0.17,"score_tight":0.13,
               "score_vol":0.08,"score_mcap":0.05,"score_phase":0.13}
    pool["composite_score"] = sum(pool[c] * w for c, w in weights.items())
    if "fresh_signal" in pool.columns:
        pool.loc[pool["fresh_signal"] == True, "composite_score"] *= 1.05
    pool = pool.sort_values("composite_score", ascending=False).reset_index(drop=True)
    pool["rank"] = pool.index + 1
    return pool

def rank_g4_pending(df):
    req = ["gate_count","g1","g2","g3","g4","g5","g6","g7"]
    if any(c not in df.columns for c in req): return pd.DataFrame()
    mask = ((df["gate_count"] == 6) & df["g1"] & df["g2"] & df["g3"] &
            (df["g4"] == False) & df["g5"] & df["g6"] & df["g7"])
    pool = df[mask].copy().reset_index(drop=True)
    if len(pool) == 0: return pool
    pool["score_vpci"]  = pool["vpci"].rank(pct=True)
    pool["score_rs"]    = pool["rs_return"].rank(pct=True)
    pool["score_52w"]   = (pool["pct_near_52w"].clip(0, 100) / 100.0)
    pool["score_tight"] = 1.0 - pool["risk_pct"].rank(pct=True)
    pool["score_vol"]   = pool["vol_ratio"].rank(pct=True) if "vol_ratio" in pool.columns else 0.5
    pool["mcap_crore"]  = pool["Market Cap"].apply(parse_mcap_to_crore)
    pool["score_mcap"]  = pool["mcap_crore"].apply(mcap_score)
    pool["score_proximity"] = pool["pct_near_52w"].clip(0, 100) / 100.0 if "pct_near_52w" in pool.columns else 0.5
    pool["score_phase"] = pool["phase"].apply(phase_score) if "phase" in pool.columns else 0.35
    weights = {"score_vpci":0.18,"score_rs":0.18,"score_52w":0.13,"score_tight":0.13,
               "score_vol":0.08,"score_mcap":0.05,"score_proximity":0.13,"score_phase":0.12}
    pool["composite_score"] = sum(pool[c] * w for c, w in weights.items())
    pool = pool.sort_values("composite_score", ascending=False).reset_index(drop=True)
    pool["rank"] = pool.index + 1
    return pool

# ════════════════════════════════ scan ════════════════════════════════
def _to_engine_df(df):
    """App fetch returns lowercase date/open/.../volume; engine wants capitalised OHLCV."""
    if df is None or len(df) == 0: return None
    d = df.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"})
    cols = [c for c in ["Open","High","Low","Close","Volume"] if c in d.columns]
    if len(cols) < 5: return None
    return d[cols].reset_index(drop=True)

def run_vpci_scan(rows, fetch_weekly_fn, relaxed=False, workers=12, progress=None):
    """rows: dicts with symbol, exch, name (+source id). fetch_weekly_fn(row)->weekly df.
    Returns (results, failed)."""
    params = {**DEFAULT_PARAMS, "relaxed": relaxed, "av_key": "demo"}
    results, failed = [], []
    def work(row):
        wk = _to_engine_df(fetch_weekly_fn(row))
        if wk is None or len(wk) < DEFAULT_PARAMS.get("vpci_long", 20) + 22:
            return row, None
        return row, analyze_stock_v3(row["symbol"], wk, params)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, r) for r in rows]
        for f in as_completed(futs):
            done += 1
            if progress: progress(done, len(rows))
            try:
                row, res = f.result()
                if res:
                    res["exch"] = row.get("exch", "NSE"); res["name"] = row.get("name", "")
                    results.append(res)
                else:
                    failed.append(None)
            except Exception:
                failed.append(None)
    return results, failed

# ════════════════════════════════ render ════════════════════════════════
def _status_label(row):
    if row.get("fresh_signal"):     return "🔥 FRESH BUY"
    if row.get("fresh_ext_signal"): return "🔥 FRESH EXT"
    if row.get("full_entry"):       return "★ BUYABLE (7/7)"
    if row.get("gates_ready_ext"):  return "⚡ 7/7 EXTENDED"
    if row.get("relaxed_entry"):    return "★ RELAXED (6/7)"
    if row.get("gate_count", 0) >= 6 and row.get("tier1_pass"): return "◉ WATCHLIST (6/7)"
    if row.get("gate_count", 0) >= 5: return "▲ MOMENTUM (5+)"
    return "Other"

def _fmt_mcap_inr(cr):
    if cr is None or (isinstance(cr, float) and np.isnan(cr)): return "N/A"
    if cr >= 1000: return f"{cr/1000:.1f}K crore inr".replace(".0K", "K")
    return f"{cr:.0f} crore inr"

def _tv(exch, sym):
    base = "https://in.tradingview.com/chart/?symbol="
    return base + ("NSE:" if exch == "NSE" else "BSE:") + sym

_TV_CFG = {"symbol": st.column_config.LinkColumn("Symbol", display_text=r".*symbol=(?:NSE:|BSE:)?(.*)")}
_SHOW = ["status","symbol","Market Cap","close","gate_count","g1","g2","g3","g4","g5","g6","g7",
         "pct_from_52w","vpci","rs_return","trail_stop","init_sl","risk_pct","vol_ratio",
         "candle_type","scenario","missed_gates"]

def _ui(df):
    d = df.copy()
    d["symbol"] = [_tv(e, s) for e, s in zip(d["exch"], d["symbol"])]
    return d[[c for c in _SHOW if c in d.columns]]

def render_vpci(results, failed, exch, get_marketcaps=None, relaxed=False):
    n_total = len(results) + len(failed)
    st.markdown("### 📊 Scan Summary")
    m1, m2, m3 = st.columns(3)
    m1.metric("Symbols analysed", n_total)
    m2.metric("✅ Candidates", len(results))
    m3.metric("⚠️ Skipped / no data", len(failed))
    if not results:
        st.warning("No candidates returned. Try a larger universe or check the data source.")
        return
    df = pd.DataFrame(results)
    df["status"] = df.apply(_status_label, axis=1)
    df_sorted = df.sort_values(["gate_count", "pct_near_52w"], ascending=[False, False]).reset_index(drop=True)

    # Market Cap only for the actionable subset (6/7 and 7/7) — keeps lookups cheap
    df_sorted["Market Cap"] = "N/A"
    if get_marketcaps is not None:
        sub = df_sorted[df_sorted["gate_count"] >= 6]
        pairs = list({(r["symbol"], r["exch"]) for _, r in sub.iterrows()})
        if pairs:
            caps = get_marketcaps(pairs)
            df_sorted["Market Cap"] = [
                _fmt_mcap_inr(caps.get((s, e))) for s, e in zip(df_sorted["symbol"], df_sorted["exch"])]
    market_flag = "NSE" if exch in ("NSE", "Both") else "BSE"

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "🔥 Fresh Signals", "★ Buyable (7/7)", "◉ Watchlist (6/7)", "All Results",
        "🏆 Ranked", "🎯 G4 Pending", "🏭 Sector Leadership", "📈 Sector Rotation"])

    with tab1:
        f = df_sorted[df_sorted["status"].isin(["🔥 FRESH BUY", "🔥 FRESH EXT"])]
        st.dataframe(_ui(f), use_container_width=True, hide_index=True, column_config=_TV_CFG) if len(f) \
            else st.info("No fresh breakout signals this week.")
    with tab2:
        b = df_sorted[df_sorted["status"] == "★ BUYABLE (7/7)"]
        st.dataframe(_ui(b), use_container_width=True, hide_index=True, column_config=_TV_CFG) if len(b) \
            else st.info("No stocks passed all 7 gates.")
    with tab3:
        w = df_sorted[df_sorted["status"].isin(["★ RELAXED (6/7)", "◉ WATCHLIST (6/7)"])]
        st.dataframe(_ui(w), use_container_width=True, hide_index=True, column_config=_TV_CFG) if len(w) \
            else st.info("No 6/7 watchlist candidates.")
    with tab4:
        st.dataframe(_ui(df_sorted), use_container_width=True, hide_index=True, column_config=_TV_CFG)

    with tab5:
        st.subheader("All Ranked Buyable Candidates")
        st.caption("Composite: VPCI 22% + RS 22% + 52wH 17% + Tight 13% + Volume 8% + Mcap 5% + Phase 13% (phase neutral here).")
        try:
            ranked = rank_stocks(df_sorted, include_relaxed=relaxed)
        except Exception as e:
            st.error(f"Ranker error: {e}"); ranked = pd.DataFrame()
        if len(ranked):
            ru = ranked.copy(); ru["symbol"] = [_tv(e, s) for e, s in zip(ru["exch"], ru["symbol"])]
            cols = ["rank","symbol","Market Cap","close","composite_score","score_vpci","score_rs",
                    "score_52w","score_tight","score_vol","score_mcap","status"]
            cols = [c for c in cols if c in ru.columns]
            st.dataframe(ru[cols].style.format({k:"{:.2f}" for k in
                ["composite_score","score_vpci","score_rs","score_52w","score_tight","score_vol","score_mcap","close"]
                if k in cols}), use_container_width=True, hide_index=True, column_config=_TV_CFG)
            st.download_button("📥 Download Ranked CSV", ranked[cols].to_csv(index=False).encode("utf-8"),
                f"vpci_ranked_{datetime.now():%Y%m%d_%H%M}.csv", "text/csv", key="vpci_rank_dl")
        else:
            st.warning("No 7/7 stocks to rank this week. Check the Watchlist tab for 6/7.")

    with tab6:
        st.subheader("🎯 G4 Pending — Pre-Breakout (6/7, only the 13-week breakout missing)")
        try:
            g4 = rank_g4_pending(df_sorted)
        except Exception as e:
            st.error(f"G4 ranker error: {e}"); g4 = pd.DataFrame()
        if len(g4):
            gu = g4.copy(); gu["symbol"] = [_tv(e, s) for e, s in zip(gu["exch"], gu["symbol"])]
            cols = ["rank","symbol","Market Cap","close","composite_score","score_vpci","score_rs",
                    "score_52w","score_tight","score_vol","score_mcap","score_proximity","status"]
            cols = [c for c in cols if c in gu.columns]
            st.dataframe(gu[cols].style.format({k:"{:.2f}" for k in
                ["composite_score","score_vpci","score_rs","score_52w","score_tight","score_vol","score_mcap","score_proximity","close"]
                if k in cols}), use_container_width=True, hide_index=True, column_config=_TV_CFG)
        else:
            st.warning("No strict G4-pending candidates this week.")

    with tab7:
        st.subheader("🏭 Leading Sectors & Industries — G5 (VPCI) ∧ G6 (RS) ∧ G7 (Above 40w)")
        smap = load_sector_map("EQUITY_L_2.csv")
        if smap.empty:
            st.warning("Sector map needs EQUITY_L_2.csv with columns: companyId, Name, Sector, Industry.")
        elif market_flag != "NSE":
            st.info("Sector leadership is available for NSE scans (the sector map covers NSE).")
        else:
            ds = attach_sector_columns(df_sorted, smap)
            lead = build_sector_leadership(ds)
            both = lead["sector_both"]
            if both.empty or both["stocks_passing"].sum() == 0:
                st.info("No stocks pass G5 ∧ G6 ∧ G7 together in this scan.")
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("Passing G5∧G6∧G7", int(both["stocks_passing"].sum()))
                c2.metric("Sectors with a leader", int((both["stocks_passing"] > 0).sum()))
                c3.metric("Top sector", both.iloc[0]["sector"], f"{int(both.iloc[0]['stocks_passing'])} stocks")
                a, b = st.columns(2)
                a.markdown("##### 🥇 Sectors (strict)"); a.dataframe(both.head(20), use_container_width=True, hide_index=True)
                b.markdown("##### 🏷️ Industries (strict)"); b.dataframe(lead["industry_both"].head(20), use_container_width=True, hide_index=True)
                with st.expander("📊 Broader breadth — G5 ∨ G6 ∨ G7"):
                    e1, e2 = st.columns(2)
                    e1.dataframe(lead["sector_either"].head(20), use_container_width=True, hide_index=True)
                    e2.dataframe(lead["industry_either"].head(20), use_container_width=True, hide_index=True)
                with st.expander("🔬 Stock-level passers"):
                    st.dataframe(lead["stock_table"], use_container_width=True, hide_index=True)
                if st.button("💾 Save weekly snapshot", key="vpci_snap"):
                    ok, msg, _ = save_weekly_snapshot(lead, market_flag)
                    (st.success if ok else st.warning)(msg)

    with tab8:
        st.subheader("📈 Sector Rotation — Week-Over-Week")
        hist = load_history_from_github() + st.session_state.get("snapshots", [])
        seen, dedup = set(), []
        for s in sorted(hist, key=lambda d: d.get("scan_date", "")):
            k = (s.get("scan_date"), s.get("market"))
            if k not in seen: seen.add(k); dedup.append(s)
        hist = [s for s in dedup if s.get("market") == market_flag]
        if not hist:
            st.info("No history yet. Open Sector Leadership and click Save snapshot each week.")
        else:
            st.success(f"{len(hist)} weekly snapshot(s) for {market_flag}.")
            rot = build_rotation_view(hist, top_n=12)
            if not rot.empty:
                st.dataframe(rot.style.background_gradient(axis=None, cmap="Greens"), use_container_width=True)
                if rot.shape[1] >= 2:
                    delta = (rot.iloc[:, -1] - rot.iloc[:, -2]).sort_values(ascending=False)
                    r1, r2 = st.columns(2)
                    r1.markdown("##### 🚀 Rotating IN"); r1.dataframe(delta[delta > 0].head(8).rename("Δ").to_frame(), use_container_width=True)
                    r2.markdown("##### 🪂 Rotating OUT"); r2.dataframe(delta[delta < 0].head(8).rename("Δ").to_frame(), use_container_width=True)
