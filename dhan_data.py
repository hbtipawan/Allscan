#!/usr/bin/env python3
"""
dhan_data.py — DhanHQ v2 data layer for the scanners.

Replaces Yahoo as the price source. Dhan uses numeric SECURITY IDs (not symbols),
so we map your NSE/BSE symbols to security IDs via Dhan's public scrip master, then
pull daily candles from POST /v2/charts/historical. Weekly candles are resampled
from daily (the historical endpoint returns daily bars).

Auth: an `access-token` (JWT) from your Dhan account with an active **Data API
subscription**. Optionally a `client-id`. Never hard-code these — pass them in
(Streamlit secrets / sidebar / env var).

Docs: https://dhanhq.co/docs/v2/historical-data/
"""
import io, json, time, urllib.request, urllib.error, datetime as dt
import numpy as np, pandas as pd

SCRIP_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"
HIST_URL  = "https://api.dhan.co/v2/charts/historical"
AUTH_URL  = "https://auth.dhan.co/app/generateAccessToken"   # headless TOTP token gen
PROFILE_URL = "https://api.dhan.co/v2/profile"               # token + Data-API status
UA = "Mozilla/5.0 (compatible; screener/1.0)"

# Nifty 50 index (for relative-strength benchmark) on Dhan
NIFTY_ROW = {"symbol": "NIFTY 50", "name": "Nifty 50", "exch": "NSE",
             "security_id": "13", "exchange_segment": "IDX_I", "instrument": "INDEX"}

# ----------------------------------------------------- auth / profile ----
def generate_token(client_id, pin, totp_secret, timeout=20):
    """Headless 24h access-token via TOTP. Requires TOTP enabled on the Dhan account.
    Computes the live 6-digit code from totp_secret, calls Dhan's generateAccessToken.
    NOTE: each successful call invalidates your previous token (call once/day)."""
    try:
        import pyotp
    except ImportError as e:
        raise RuntimeError("pyotp is not installed (add 'pyotp' to requirements.txt).") from e
    code = pyotp.TOTP(str(totp_secret).strip().replace(" ", "")).now()
    url = (f"{AUTH_URL}?dhanClientId={str(client_id).strip()}"
           f"&pin={str(pin).strip()}&totp={code}")
    req = urllib.request.Request(url, data=b"", method="POST",
                                 headers={"Accept": "application/json", "User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.load(r)
    except urllib.error.HTTPError as e:
        try: msg = json.load(e).get("message") or json.load(e).get("errorMessage")
        except Exception: msg = f"HTTP {e.code}"
        raise PermissionError(f"Dhan token generation failed: {msg}. Check Client ID, PIN, "
                              f"and that TOTP is enabled with the correct secret.") from e
    tok = d.get("accessToken") or d.get("access_token")
    if not tok:
        raise PermissionError(f"Dhan token generation failed: {d.get('message') or d}. "
                              f"Ensure TOTP is enabled and Client ID/PIN/secret are correct.")
    return tok, d.get("expiryTime") or d.get("expiry_time")

def get_profile(token, client_id=None, timeout=20):
    """GET /v2/profile -> dict with tokenValidity, dataPlan (Data-API status), dataValidity,
    activeSegment, etc. Used to diagnose auth/subscription problems."""
    headers = {"Accept": "application/json", "access-token": token or "", "User-Agent": UA}
    if client_id: headers["client-id"] = str(client_id)
    req = urllib.request.Request(PROFILE_URL, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        try: return json.load(e)
        except Exception: return {"errorMessage": f"HTTP {e.code}"}
    except Exception as e:
        return {"errorMessage": str(e)}

# --------------------------------------------------------- scrip master ----
def load_scrip_master(url=SCRIP_URL):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=90) as r:
        raw = r.read()
    return pd.read_csv(io.BytesIO(raw), low_memory=False)

def build_symbol_maps(scrip):
    """Return (nse_map, bse_map): {UPPER_SYMBOL: (security_id, exchange_segment)}.
    NSE uses the EQ series (main board); BSE uses any equity row."""
    eq = scrip[scrip["SEM_INSTRUMENT_NAME"].astype(str).str.strip() == "EQUITY"].copy()
    eq["sym"] = eq["SEM_TRADING_SYMBOL"].astype(str).str.strip().str.upper()
    eq["ser"] = eq["SEM_SERIES"].astype(str).str.strip()
    def to_map(d, seg):
        m = {}
        for sym, sid in zip(d["sym"], d["SEM_SMST_SECURITY_ID"]):
            if sym and sym not in m and pd.notna(sid):
                try: m[sym] = (str(int(sid)), seg)
                except Exception: pass
        return m
    nse = eq[eq["SEM_EXM_EXCH_ID"] == "NSE"]
    nse = pd.concat([nse[nse["ser"] == "EQ"], nse[nse["ser"] != "EQ"]])  # prefer EQ series
    bse = eq[eq["SEM_EXM_EXCH_ID"] == "BSE"]
    return to_map(nse, "NSE_EQ"), to_map(bse, "BSE_EQ")

def map_universe(uni, nse_map, bse_map):
    """uni: DataFrame with columns symbol, exch (+name,sector). Adds security_id,
    exchange_segment, instrument. Drops rows with no Dhan mapping."""
    sid, seg = [], []
    for s, e in zip(uni["symbol"].astype(str).str.upper(), uni["exch"]):
        hit = (nse_map if e == "NSE" else bse_map).get(s)
        sid.append(hit[0] if hit else None); seg.append(hit[1] if hit else None)
    out = uni.copy()
    out["security_id"] = sid; out["exchange_segment"] = seg; out["instrument"] = "EQUITY"
    mapped = out[out["security_id"].notna()].reset_index(drop=True)
    unmapped = sorted(out[out["security_id"].isna()]["symbol"].tolist())
    return mapped, unmapped

def full_universe(exch="NSE", scrip=None):
    """Build the ENTIRE NSE/BSE cash-equity universe from Dhan's scrip master.
    Returns DataFrame[symbol,name,sector,exch,security_id,exchange_segment,instrument]."""
    scrip = scrip if scrip is not None else load_scrip_master()
    eq = scrip[scrip["SEM_INSTRUMENT_NAME"].astype(str).str.strip() == "EQUITY"].copy()
    eq["sym"] = eq["SEM_TRADING_SYMBOL"].astype(str).str.strip().str.upper()
    eq["ser"] = eq["SEM_SERIES"].astype(str).str.strip()
    nm_col = "SM_SYMBOL_NAME" if "SM_SYMBOL_NAME" in eq.columns else "SEM_CUSTOM_SYMBOL"
    rows = []
    def add(df, exch_id, seg):
        seen = set()
        for sym, sid, nm in zip(df["sym"], df["SEM_SMST_SECURITY_ID"], df[nm_col]):
            if sym and sym not in seen and pd.notna(sid):
                seen.add(sym)
                rows.append(dict(symbol=sym, name=str(nm), sector="", exch=exch_id,
                                 security_id=str(int(sid)), exchange_segment=seg, instrument="EQUITY"))
    if exch in ("NSE", "Both"):
        nse = eq[eq["SEM_EXM_EXCH_ID"] == "NSE"]
        nse = pd.concat([nse[nse["ser"] == "EQ"], nse[nse["ser"] != "EQ"]])
        add(nse, "NSE", "NSE_EQ")
    if exch in ("BSE", "Both"):
        add(eq[eq["SEM_EXM_EXCH_ID"] == "BSE"], "BSE", "BSE_EQ")
    return pd.DataFrame(rows)

# --------------------------------------------------------- daily fetch ----
def fetch_daily(security_id, exchange_segment, instrument, from_date, to_date,
                token, client_id=None, retries=2):
    body = json.dumps({"securityId": str(security_id), "exchangeSegment": exchange_segment,
        "instrument": instrument, "expiryCode": 0, "oi": False,
        "fromDate": from_date, "toDate": to_date}).encode()
    headers = {"Content-Type": "application/json", "Accept": "application/json",
               "access-token": token or ""}
    if client_id: headers["client-id"] = str(client_id)
    last = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(HIST_URL, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=30) as r:
                d = json.load(r)
            if not isinstance(d, dict) or not d.get("close"):
                return None
            df = pd.DataFrame({
                "date": pd.to_datetime(d["timestamp"], unit="s").normalize(),
                "open": d["open"], "high": d["high"], "low": d["low"],
                "close": d["close"], "volume": d.get("volume", [0]*len(d["close"]))})
            return df.dropna(subset=["close"]).reset_index(drop=True)
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}"
            if e.code == 429:                       # rate limited -> back off and retry
                time.sleep(1.5 * (attempt + 1)); continue
            if e.code in (401, 403):                # auth/subscription -> don't retry
                raise PermissionError("Dhan auth/subscription error (check access-token "
                                      "and Data API subscription).") from e
            time.sleep(0.5 * (attempt + 1))
        except Exception as e:
            last = str(e); time.sleep(0.4 * (attempt + 1))
    return None

def to_weekly(df):
    if df is None or df.empty: return df
    s = df.set_index("date").sort_index()
    w = s.resample("W-FRI").agg({"open": "first", "high": "max", "low": "min",
                                 "close": "last", "volume": "sum"}).dropna(subset=["close"])
    return w.reset_index()

def make_fetch_fn(token, weekly=False, years=2.0, client_id=None):
    """Returns fetch_fn(row)->DataFrame|None. row needs security_id, exchange_segment, instrument."""
    to_d = dt.date.today()
    from_d = to_d - dt.timedelta(days=int(years * 365) + 15)
    fs, ts = from_d.isoformat(), to_d.isoformat()
    def f(row):
        df = fetch_daily(row["security_id"], row["exchange_segment"],
                         row.get("instrument", "EQUITY"), fs, ts, token, client_id)
        return to_weekly(df) if (weekly and df is not None) else df
    return f
