"""
sector_history.py
─────────────────
Three-in-one professional helper module for the VPCI screener:

  1. Sector / Industry mapping   — load NSE master and join to scan results.
  2. Weekly snapshot persistence — push each scan's leadership to GitHub
                                   so sector rotation can be tracked over weeks.
  3. Scan checkpoint recovery    — periodic dump of in-progress results to
                                   st.session_state + downloadable JSON, so a
                                   network drop doesn't force a full re-scan.

Designed to be additive — does not touch any existing scoring / gate logic.
"""

from __future__ import annotations

import base64
import json
import os
from datetime import datetime, date
from typing import Optional

import pandas as pd
import requests
import streamlit as st


# ═══════════════════════════════════════════════════════════════════════
# 1.  SECTOR / INDUSTRY MAPPING
# ═══════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_sector_map(path: str = "EQUITY_L_2.csv") -> pd.DataFrame:
    """
    Load the NSE master CSV and return a DataFrame indexed by symbol with
    Sector / Industry / Name columns. Handles both the new schema
    (companyId, Name, Sector, Industry) and the legacy schema (SYMBOL, NAME...).
    Returns an empty mapping with the right columns on any failure so the rest
    of the app degrades gracefully instead of crashing.
    """
    try:
        df = pd.read_csv(path)
        df.columns = [str(c).strip() for c in df.columns]

        # New schema (preferred)
        if {"companyId", "Sector", "Industry"}.issubset(df.columns):
            out = df.rename(columns={
                "companyId": "symbol",
                "Name": "name",
                "Sector": "sector",
                "Industry": "industry",
            })[["symbol", "name", "sector", "industry"]].copy()
        # Legacy schema
        elif "SYMBOL" in df.columns:
            out = pd.DataFrame({
                "symbol":   df["SYMBOL"].astype(str).str.strip(),
                "name":     df.get("NAME OF COMPANY", df.get("Name", "")),
                "sector":   "Unknown",
                "industry": "Unknown",
            })
        else:
            return pd.DataFrame(columns=["symbol", "name", "sector", "industry"])

        out["symbol"]   = out["symbol"].astype(str).str.strip().str.upper()
        out["sector"]   = out["sector"].fillna("Unknown").astype(str).str.strip()
        out["industry"] = out["industry"].fillna("Unknown").astype(str).str.strip()
        return out.drop_duplicates(subset=["symbol"]).reset_index(drop=True)

    except Exception as e:
        st.warning(f"Sector map could not be loaded ({e}). Sector tab will be empty.")
        return pd.DataFrame(columns=["symbol", "name", "sector", "industry"])


def attach_sector_columns(df_scan: pd.DataFrame, sector_map: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join sector / industry / company-name onto scan results.
    The scan DataFrame's symbol column is matched case-insensitively.
    Original DataFrame is not mutated.
    """
    if df_scan is None or df_scan.empty or sector_map.empty:
        out = df_scan.copy() if df_scan is not None else pd.DataFrame()
        for col in ("sector", "industry", "name"):
            if col not in out.columns:
                out[col] = "Unknown"
        return out

    out = df_scan.copy()
    out["_join_key"] = out["symbol"].astype(str).str.strip().str.upper()
    smap = sector_map.set_index("symbol")[["sector", "industry", "name"]]
    out = out.join(smap, on="_join_key")
    out[["sector", "industry", "name"]] = out[["sector", "industry", "name"]].fillna("Unknown")
    return out.drop(columns="_join_key")


def build_sector_leadership(df_with_sector: pd.DataFrame) -> dict:
    """
    Given a scan DataFrame that has g5, g6, g7 boolean columns and
    sector/industry columns, produce the leadership tables:

      • Sectors ranked by count of stocks where (g5 AND g6 AND g7) — the
        strict institutional-confirmation cut:
            G5 = VPCI accumulation,
            G6 = Leader RS positive vs index,
            G7 = Price above 40-week MA (Stage 2 trend).
        When several constituents of the same sector trip all three together,
        smart money, relative strength, and trend are aligned — the highest-
        probability sector to allocate to.
      • Industries ranked the same way.
      • A broader 'breadth' cut using (g5 OR g6 OR g7) — useful for spotting
        sectors just starting to rotate, where only one or two of the three
        gates are firing yet.

    Returns a dict of DataFrames so the caller can render each table in the UI.
    """
    empty = pd.DataFrame(columns=["sector", "stocks_passing", "total_in_universe", "hit_rate_%"])
    if df_with_sector is None or df_with_sector.empty:
        return {"sector_both": empty, "industry_both": empty,
                "sector_either": empty, "industry_either": empty,
                "stock_table": pd.DataFrame()}

    df = df_with_sector.copy()
    for c in ("g5", "g6", "g7"):
        if c not in df.columns:
            df[c] = False
    df["g5"] = df["g5"].astype(bool)
    df["g6"] = df["g6"].astype(bool)
    df["g7"] = df["g7"].astype(bool)
    # Naming kept as 'both' / 'either' for backwards compatibility with the
    # snapshot JSON schema; semantics now mean "all three" / "any of three".
    df["both"]   = df["g5"] & df["g6"] & df["g7"]
    df["either"] = df["g5"] | df["g6"] | df["g7"]

    def _agg(group_col: str, flag_col: str) -> pd.DataFrame:
        total  = df.groupby(group_col).size().rename("total_in_universe")
        passing = df[df[flag_col]].groupby(group_col).size().rename("stocks_passing")
        out = pd.concat([passing, total], axis=1).fillna(0).astype(int)
        out["hit_rate_%"] = (out["stocks_passing"] / out["total_in_universe"] * 100).round(1)
        out = out.reset_index().rename(columns={group_col: group_col})
        out = out.sort_values(["stocks_passing", "hit_rate_%"], ascending=[False, False])
        return out.reset_index(drop=True)

    stock_cols_full = ["symbol", "name", "sector", "industry",
                       "g5", "g6", "g7", "gate_count"]
    stock_cols_min  = ["symbol", "name", "sector", "industry", "g5", "g6", "g7"]

    return {
        "sector_both":     _agg("sector",   "both"),
        "industry_both":   _agg("industry", "both"),
        "sector_either":   _agg("sector",   "either"),
        "industry_either": _agg("industry", "either"),
        "stock_table":     df[df["both"]][stock_cols_full]
                            .sort_values(["sector", "industry", "symbol"])
                            .reset_index(drop=True)
                          if "gate_count" in df.columns
                          else df[df["both"]][stock_cols_min]
                                .sort_values(["sector", "industry", "symbol"])
                                .reset_index(drop=True),
    }


# ═══════════════════════════════════════════════════════════════════════
# 2.  WEEKLY SNAPSHOT PERSISTENCE  (GitHub-backed, optional)
# ═══════════════════════════════════════════════════════════════════════
#
# Streamlit Community Cloud's filesystem is ephemeral — anything written
# to disk is wiped on container restart. To track sector rotation across
# weeks we push small JSON snapshots to a GitHub repo using a Personal
# Access Token stored in st.secrets. If no token is configured we silently
# fall back to in-memory snapshots (visible during the session only).
#
# Required secret in `.streamlit/secrets.toml` :
#
#   [github]
#   token  = "ghp_xxxxxxxxxxxxxxxxxx"
#   repo   = "hbtipawan/Screener-Back-UP"
#   branch = "main"
#   folder = "history"            # folder inside the repo to write to
#
# Snapshot file name pattern: history/sector_YYYY-MM-DD.json
# ═══════════════════════════════════════════════════════════════════════

def _gh_secrets() -> Optional[dict]:
    try:
        s = st.secrets["github"]
        return {
            "token":  s["token"],
            "repo":   s["repo"],
            "branch": s.get("branch", "main"),
            "folder": s.get("folder", "history"),
        }
    except Exception:
        return None


def _gh_put_file(path_in_repo: str, content_str: str, commit_msg: str) -> tuple[bool, str]:
    """Create or update a file in the configured GitHub repo. Returns (ok, msg)."""
    cfg = _gh_secrets()
    if not cfg:
        return False, "No [github] section in secrets.toml — snapshot kept in session only."

    api = f"https://api.github.com/repos/{cfg['repo']}/contents/{path_in_repo}"
    headers = {
        "Authorization": f"Bearer {cfg['token']}",
        "Accept": "application/vnd.github+json",
    }
    # Check whether file exists to capture sha for an update
    sha = None
    try:
        r = requests.get(api, headers=headers, params={"ref": cfg["branch"]}, timeout=15)
        if r.status_code == 200:
            sha = r.json().get("sha")
    except Exception:
        pass

    payload = {
        "message": commit_msg,
        "content": base64.b64encode(content_str.encode("utf-8")).decode("ascii"),
        "branch":  cfg["branch"],
    }
    if sha:
        payload["sha"] = sha

    try:
        r = requests.put(api, headers=headers, data=json.dumps(payload), timeout=20)
        if r.status_code in (200, 201):
            return True, f"Snapshot pushed to {cfg['repo']}/{path_in_repo}"
        return False, f"GitHub API {r.status_code}: {r.text[:200]}"
    except Exception as e:
        return False, f"GitHub push failed: {e}"


def _gh_list_history() -> list[dict]:
    """List snapshot files in the configured history folder."""
    cfg = _gh_secrets()
    if not cfg:
        return []
    api = f"https://api.github.com/repos/{cfg['repo']}/contents/{cfg['folder']}"
    headers = {
        "Authorization": f"Bearer {cfg['token']}",
        "Accept": "application/vnd.github+json",
    }
    try:
        r = requests.get(api, headers=headers, params={"ref": cfg["branch"]}, timeout=15)
        if r.status_code != 200:
            return []
        items = r.json()
        return [
            {"name": it["name"], "download_url": it["download_url"]}
            for it in items
            if isinstance(it, dict) and it.get("name", "").startswith("sector_")
            and it.get("name", "").endswith(".json")
        ]
    except Exception:
        return []


def save_weekly_snapshot(leadership: dict, market: str, scan_date: Optional[date] = None
                         ) -> tuple[bool, str, dict]:
    """
    Persist the current week's sector leadership tables. Returns
    (ok, message, snapshot_dict). The snapshot is also kept in
    st.session_state['snapshots'] regardless of GitHub success.
    """
    scan_date = scan_date or date.today()
    snap = {
        "scan_date":      scan_date.isoformat(),
        "scan_timestamp": datetime.now().isoformat(timespec="seconds"),
        "market":         market,
        "sector_both":     leadership["sector_both"].to_dict(orient="records"),
        "industry_both":   leadership["industry_both"].to_dict(orient="records"),
        "sector_either":   leadership["sector_either"].to_dict(orient="records"),
        "industry_either": leadership["industry_either"].to_dict(orient="records"),
    }

    # Always keep an in-session copy
    snaps = st.session_state.setdefault("snapshots", [])
    snaps.append(snap)
    st.session_state["snapshots"] = snaps[-26:]  # keep last 6 months max

    cfg = _gh_secrets()
    if not cfg:
        return False, "Saved to session only (configure GitHub secrets to persist across weeks).", snap

    fname = f"{cfg['folder']}/sector_{scan_date.isoformat()}_{market}.json"
    ok, msg = _gh_put_file(
        fname,
        json.dumps(snap, indent=2),
        f"Weekly snapshot {scan_date.isoformat()} ({market})",
    )
    return ok, msg, snap


def load_history_from_github() -> list[dict]:
    """Pull all weekly snapshots from GitHub for trend analysis."""
    items = _gh_list_history()
    history = []
    for it in items:
        try:
            r = requests.get(it["download_url"], timeout=15)
            if r.status_code == 200:
                history.append(r.json())
        except Exception:
            continue
    history.sort(key=lambda d: d.get("scan_date", ""))
    return history


def build_rotation_view(history: list[dict], top_n: int = 10) -> pd.DataFrame:
    """
    Build a wide DataFrame: rows = sectors, columns = scan dates,
    cells  = stocks_passing (g6 AND g7). Limited to the union of the top_n
    sectors from each week so the table stays readable.
    """
    if not history:
        return pd.DataFrame()

    keep = set()
    for snap in history:
        for row in snap.get("sector_both", [])[:top_n]:
            keep.add(row["sector"])

    frames = []
    for snap in history:
        d = snap.get("scan_date", "?")
        df = pd.DataFrame(snap.get("sector_both", []))
        if df.empty:
            continue
        df = df[df["sector"].isin(keep)][["sector", "stocks_passing"]]
        df = df.rename(columns={"stocks_passing": d}).set_index("sector")
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1).fillna(0).astype(int)
    out["latest"] = out.iloc[:, -1]
    return out.sort_values("latest", ascending=False).drop(columns="latest")


# ═══════════════════════════════════════════════════════════════════════
# 3.  SCAN CHECKPOINTING  (resume-on-disconnect)
# ═══════════════════════════════════════════════════════════════════════
#
# We checkpoint partial scan progress to st.session_state every CHECKPOINT_EVERY
# completed symbols. This survives:
#   • Streamlit reruns triggered by widget interaction
#   • Browser tab reload (within the SAME session id, depending on platform)
#   • Brief network blips that don't kill the WebSocket
#
# It does NOT survive Streamlit Cloud container restarts — for that the user
# can download the checkpoint JSON and re-upload it next time.
# ═══════════════════════════════════════════════════════════════════════

CHECKPOINT_EVERY = 25


def _ckpt_key(market: str) -> str:
    return f"scan_ckpt::{market}"


def init_checkpoint(market: str, all_symbols: list[str]) -> tuple[list[dict], list[str], list[str]]:
    """
    Return (results_so_far, failed_so_far, remaining_symbols).
    If a usable checkpoint exists for this market AND the symbol universe
    matches, resume from where we stopped; otherwise start fresh.
    """
    key = _ckpt_key(market)
    ckpt = st.session_state.get(key)
    if not ckpt or ckpt.get("market") != market:
        return [], [], list(all_symbols)

    # Sanity check: universe must be the same (same length + same first/last)
    saved_universe = ckpt.get("universe", [])
    if (len(saved_universe) != len(all_symbols)
            or (all_symbols and saved_universe
                and (saved_universe[0] != all_symbols[0]
                     or saved_universe[-1] != all_symbols[-1]))):
        return [], [], list(all_symbols)

    done_syms = set(ckpt.get("done_symbols", []))
    remaining = [s for s in all_symbols if s not in done_syms]
    return list(ckpt.get("results", [])), list(ckpt.get("failed", [])), remaining


def save_checkpoint(market: str, all_symbols: list[str],
                    results: list[dict], failed: list[str],
                    done_symbols: list[str]) -> None:
    st.session_state[_ckpt_key(market)] = {
        "market":       market,
        "saved_at":     datetime.now().isoformat(timespec="seconds"),
        "universe":     all_symbols,
        "results":      results,
        "failed":       failed,
        "done_symbols": done_symbols,
    }


def clear_checkpoint(market: str) -> None:
    st.session_state.pop(_ckpt_key(market), None)


def checkpoint_to_json(market: str) -> Optional[bytes]:
    """Serialize the current checkpoint so the user can download it."""
    ckpt = st.session_state.get(_ckpt_key(market))
    if not ckpt:
        return None
    return json.dumps(ckpt, default=str, indent=2).encode("utf-8")


def restore_checkpoint_from_upload(uploaded_bytes: bytes) -> tuple[bool, str]:
    """Restore a previously-downloaded checkpoint JSON into session state."""
    try:
        ckpt = json.loads(uploaded_bytes.decode("utf-8"))
        market = ckpt.get("market")
        if not market:
            return False, "Uploaded file is missing 'market' field."
        st.session_state[_ckpt_key(market)] = ckpt
        return True, (f"Checkpoint restored for {market} — "
                      f"{len(ckpt.get('done_symbols', []))} symbols already scanned.")
    except Exception as e:
        return False, f"Could not parse checkpoint: {e}"
