#!/usr/bin/env python3
"""
============================================================================
 screener_core.py  —  Bullish reversal screener ENGINE (single source of truth)
 Imported by both the Streamlit app (streamlit_app.py) and the CLI
 (bullish_reversal_screener.py).  Detection logic is identical everywhere.

 Data: Yahoo Finance v8 chart API.  NSE -> "<SYMBOL>.NS", BSE -> "<SCRIP_ID>.BO".
 Deps: pandas, numpy.  (Streamlit only needed by the app, not by this engine.)

 EVERY detector enforces the LOCATION GATE: a bullish reversal is only valid
 after a prior DOWNTREND (and, for single-candle signals, near a recent LOW).
============================================================================
"""
import sys, json, math, time, urllib.request, urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np, pandas as pd

def tv_url(exch, symbol):
    """TradingView chart URL, e.g. NSE:KANSAINER / BSE:3BFILMS."""
    ex = (exch or "NSE").upper()
    return f"https://www.tradingview.com/chart/?symbol={ex}:{urllib.parse.quote(str(symbol))}"

# ----------------------------------------------------------------- config ----
RANGE     = "1y"     # history per symbol (~250 daily bars)
MIN_BARS  = 40       # need this many bars for indicators
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120 Safari/537.36")
HOSTS = ["query1.finance.yahoo.com", "query2.finance.yahoo.com"]

# ------------------------------------------------------------------ fetch ----
def fetch_ohlcv(yahoo_symbol, rng=RANGE, interval="1d", retries=2):
    """yahoo_symbol includes the suffix, e.g. 'RELIANCE.NS' / '3BFILMS.BO'.
    interval: '1d' (daily) or '1wk' (weekly)."""
    sym = yahoo_symbol.replace("&", "%26")
    last = None
    for attempt in range(retries+1):
        for host in HOSTS:
            url = f"https://{host}/v8/finance/chart/{sym}?range={rng}&interval={interval}"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=25) as resp:
                    d = json.load(resp)
                r = d["chart"]["result"][0]
                q = r["indicators"]["quote"][0]
                df = pd.DataFrame({
                    "date": pd.to_datetime(r["timestamp"], unit="s").normalize(),
                    "open": q["open"], "high": q["high"], "low": q["low"],
                    "close": q["close"], "volume": q["volume"]})
                df = df.dropna().reset_index(drop=True)
                return df if len(df) else None
            except Exception as e:
                last = e
        time.sleep(0.6*(attempt+1))   # backoff on rate-limit
    return None

# ------------------------------------------------------------- indicators ----
def make_arrays(df):
    o=df["open"].values.astype(float);  h=df["high"].values.astype(float)
    l=df["low"].values.astype(float);   c=df["close"].values.astype(float)
    v=df["volume"].values.astype(float)
    body=np.abs(c-o); rng=h-l
    upsh=h-np.maximum(o,c); dnsh=np.minimum(o,c)-l
    def sma(x,n): return pd.Series(x).rolling(n).mean().values
    def ema(x,n): return pd.Series(x).ewm(span=n,adjust=False).mean().values
    pc=np.r_[np.nan, c[:-1]]
    tr=np.maximum.reduce([h-l, np.abs(h-pc), np.abs(l-pc)])
    atr=pd.Series(tr).ewm(alpha=1/14,adjust=False).mean().values
    return dict(o=o,h=h,l=l,c=c,v=v,body=body,rng=rng,upsh=upsh,dnsh=dnsh,
        bull=c>o, bear=c<o,
        body_sma=sma(body,14), rng_sma=sma(rng,14), atr=atr,
        ema20=ema(c,20), ema10=ema(c,10), vol_sma=sma(v,20),
        low_min10=pd.Series(l).rolling(10).min().values,
        low_min10_prev=pd.Series(l).shift(1).rolling(10).min().values,
        n=len(c))

# ------------------------------------------------------------------ gates ----
def ok(x):           return x is not None and not (isinstance(x,float) and math.isnan(x))
def long_body(A,i):  return ok(A['body_sma'][i]) and A['body_sma'][i]>0 and A['body'][i] > 1.3*A['body_sma'][i]
def small_body(A,i): return ok(A['body_sma'][i]) and A['body_sma'][i]>0 and A['body'][i] < 0.6*A['body_sma'][i]
def doji(A,i):       return A['rng'][i]>0 and A['body'][i] <= 0.10*A['rng'][i]
def near_low(A,i):
    m=A['low_min10'][i]; return ok(m) and A['l'][i] <= m*1.01
def downtrend(A,r):
    if r-5 < 0 or not ok(A['ema20'][r]): return False
    return (A['c'][r] < A['ema20'][r]) and (A['c'][r] < A['c'][r-5])
def vol_conf(A,i):   return ok(A['vol_sma'][i]) and A['vol_sma'][i]>0 and A['v'][i] >= 1.5*A['vol_sma'][i]
def midbody(A,i):    return (A['o'][i]+A['c'][i])/2.0

# -------------------------------------------------------------- detectors ----
def d_hammer(A,i):
    if i<MIN_BARS: return None
    r=A['rng'][i]; b=A['body'][i]
    if r<=0 or b<=0.05*r or b>0.40*r: return None
    if A['dnsh'][i] < 2*b or A['upsh'][i] > 0.30*r: return None
    if not near_low(A,i) or not downtrend(A,i-1): return None
    return dict(stop=A['l'][i], note=f"Lower shadow {A['dnsh'][i]/max(b,1e-9):.1f}x body; lows rejected.")

def d_inverted_hammer(A,i):
    if i<MIN_BARS: return None
    r=A['rng'][i]; b=A['body'][i]
    if r<=0 or b<=0.05*r or b>0.40*r: return None
    if A['upsh'][i] < 2*b or A['dnsh'][i] > 0.30*r: return None
    if not near_low(A,i) or not downtrend(A,i-1): return None
    return dict(stop=A['l'][i], note="Long upper shadow at a low; needs up-close confirm next bar.")

def d_dragonfly(A,i):
    if i<MIN_BARS: return None
    r=A['rng'][i]
    if r<=0 or not doji(A,i): return None
    if A['dnsh'][i] < 0.60*r or A['upsh'][i] > 0.10*r: return None
    if not near_low(A,i) or not downtrend(A,i-1): return None
    return dict(stop=A['l'][i], note="Dragonfly doji at support; indecision with bullish lean.")

def d_bull_engulf(A,i):
    if i<MIN_BARS+1: return None
    if not (A['bear'][i-1] and A['bull'][i]): return None
    if not (A['o'][i] <= A['c'][i-1] and A['c'][i] >= A['o'][i-1]): return None
    if not (A['o'][i] < A['c'][i-1] or A['c'][i] > A['o'][i-1]): return None
    if A['body'][i] <= A['body'][i-1]: return None
    if not downtrend(A,i-2): return None
    return dict(stop=min(A['l'][i-1],A['l'][i]), note="Green body fully engulfs prior red body.")

def d_piercing(A,i):
    if i<MIN_BARS+1: return None
    if not (A['bear'][i-1] and long_body(A,i-1) and A['bull'][i]): return None
    if not (A['o'][i] < A['c'][i-1]): return None
    if not (A['c'][i] > midbody(A,i-1) and A['c'][i] < A['o'][i-1]): return None
    if not downtrend(A,i-2): return None
    return dict(stop=min(A['l'][i-1],A['l'][i]), note="Closes above midpoint of prior long red.")

def d_tweezer_bottom(A,i):
    if i<MIN_BARS+1: return None
    tol = 0.0025*A['c'][i]          # lows must match within ~0.25% of price (tight)
    if abs(A['l'][i-1]-A['l'][i]) > tol: return None
    if not A['bull'][i]: return None
    if not near_low(A,i) or not downtrend(A,i-2): return None
    return dict(stop=min(A['l'][i-1],A['l'][i]), note="Two candles reject the same low.")

def d_morning_star(A,i):
    if i<MIN_BARS+2: return None
    c1,c2,c3=i-2,i-1,i
    if not (A['bear'][c1] and long_body(A,c1)): return None
    if A['body'][c2] >= 0.5*A['body'][c1]: return None
    if max(A['o'][c2],A['c'][c2]) > A['c'][c1]*1.005: return None
    if not (A['bull'][c3] and A['c'][c3] > midbody(A,c1)): return None
    if not downtrend(A,c1-1): return None
    return dict(stop=min(A['l'][c1],A['l'][c2],A['l'][c3]), note="Star low + strong close into the red body.")

def d_three_white(A,i):
    if i<MIN_BARS+2: return None
    a,b,cc=i-2,i-1,i
    if not (A['bull'][a] and A['bull'][b] and A['bull'][cc]): return None
    if not (A['c'][a]<A['c'][b]<A['c'][cc]): return None
    if not (min(A['o'][a],A['c'][a]) <= A['o'][b] <= max(A['o'][a],A['c'][a])): return None
    if not (min(A['o'][b],A['c'][b]) <= A['o'][cc] <= max(A['o'][b],A['c'][b])): return None
    for j in (a,b,cc):
        if A['body'][j] <= 0 or A['upsh'][j] > 0.35*A['body'][j]: return None
        if not (A['body_sma'][j]>0 and A['body'][j] > 0.5*A['body_sma'][j]): return None
    if not downtrend(A,a-1): return None
    return dict(stop=min(A['l'][a],A['l'][b],A['l'][cc]), note="Three rising soldiers off a low.")

def d_three_inside_up(A,i):
    if i<MIN_BARS+2: return None
    c1,c2,c3=i-2,i-1,i
    if not (A['bear'][c1] and long_body(A,c1)): return None
    if not (A['bull'][c2] and A['o'][c2] > A['c'][c1] and A['c'][c2] < A['o'][c1]): return None
    if not (A['bull'][c3] and A['c'][c3] > A['o'][c1]): return None
    if not downtrend(A,c1-1): return None
    return dict(stop=min(A['l'][c1],A['l'][c2],A['l'][c3]), note="Bullish harami + up-close confirmation.")

def d_three_outside_up(A,i):
    if i<MIN_BARS+2: return None
    c1,c2,c3=i-2,i-1,i
    if not (A['bear'][c1] and A['bull'][c2]): return None
    if not (A['o'][c2] <= A['c'][c1] and A['c'][c2] >= A['o'][c1] and A['body'][c2]>A['body'][c1]): return None
    if not (A['bull'][c3] and A['c'][c3] > A['c'][c2]): return None
    if not downtrend(A,c1-1): return None
    return dict(stop=min(A['l'][c1],A['l'][c2],A['l'][c3]), note="Engulfing + higher confirmation close.")

def d_three_line_strike(A,i):
    if i<MIN_BARS+3: return None
    a,b,c,d=i-3,i-2,i-1,i
    if not (A['bear'][a] and A['bear'][b] and A['bear'][c]): return None
    if not (A['c'][a] > A['c'][b] > A['c'][c]): return None
    if not (A['bull'][d] and A['o'][d] <= A['c'][c] and A['c'][d] >= A['o'][a]): return None
    if not downtrend(A,a-1): return None
    return dict(stop=min(A['l'][a],A['l'][b],A['l'][c],A['l'][d]),
                note="Big green erases three red candles (Bulkowski's top reversal).")

def d_abandoned_baby(A,i):
    if i<MIN_BARS+2: return None
    c1,c2,c3=i-2,i-1,i
    if not (A['bear'][c1] and doji(A,c2) and A['bull'][c3]): return None
    if not (A['h'][c2] < A['l'][c1]): return None
    if not (A['l'][c3] > A['h'][c2]): return None
    if not downtrend(A,c1-1): return None
    return dict(stop=min(A['l'][c1],A['l'][c2],A['l'][c3]), note="Island-reversal doji, both gaps clean.")

def d_bull_harami(A,i):
    if i<MIN_BARS+1: return None
    c1,c2=i-1,i
    if not (A['bear'][c1] and long_body(A,c1)): return None
    if not (A['bull'][c2] and A['o'][c2] > A['c'][c1] and A['c'][c2] < A['o'][c1]): return None
    if A['body'][c2] >= 0.6*A['body'][c1]: return None
    if not downtrend(A,c1-1): return None
    return dict(stop=min(A['l'][c1],A['l'][c2]), note="Small green inside a long red - momentum warning.")

def d_selling_climax(A,i):
    if i<MIN_BARS: return None
    if not (ok(A['rng_sma'][i]) and A['rng'][i] >= 1.5*A['rng_sma'][i]): return None
    if not (ok(A['vol_sma'][i]) and A['vol_sma'][i]>0 and A['v'][i] >= 2.0*A['vol_sma'][i]): return None
    if not (A['bear'][i] or (A['rng'][i]>0 and A['dnsh'][i] >= 0.40*A['rng'][i])): return None
    if A['rng'][i]>0 and A['upsh'][i] > 0.35*A['rng'][i]: return None   # reject upthrust (upper-wick dominated)
    if not near_low(A,i) or not downtrend(A,i-1): return None
    return dict(stop=A['l'][i], note=f"Wide bar on {A['v'][i]/A['vol_sma'][i]:.1f}x volume - capitulation.")

def d_stopping_volume(A,i):
    if i<MIN_BARS: return None
    r=A['rng'][i]
    if r<=0: return None
    if not (ok(A['vol_sma'][i]) and A['vol_sma'][i]>0 and A['v'][i] >= 1.8*A['vol_sma'][i]): return None
    if A['dnsh'][i] < 0.50*r: return None
    if A['c'][i] < A['l'][i] + 0.5*r: return None
    if not (ok(A['low_min10_prev'][i]) and A['l'][i] < A['low_min10_prev'][i]): return None
    if not downtrend(A,i-1): return None
    return dict(stop=A['l'][i], note=f"New low absorbed on {A['v'][i]/A['vol_sma'][i]:.1f}x volume; close off lows.")

def d_no_supply(A,i):
    if i<MIN_BARS: return None
    if not A['bear'][i]: return None
    if not (ok(A['rng_sma'][i]) and A['rng'][i] <= 0.70*A['rng_sma'][i]): return None
    if not (ok(A['vol_sma'][i]) and A['v'][i] <= 0.70*A['vol_sma'][i]): return None
    if not (A['v'][i] < A['v'][i-1] and A['v'][i] < A['v'][i-2]): return None
    if not downtrend(A,i-1): return None
    return dict(stop=A['l'][i], note="Narrow down-bar on dried-up volume - no sellers left.")

# ------------------------------------------------------ pattern registry ----
# (name, win_prob, sourced?, tier, detector, pattern_length, literature one-liner)
PATTERNS = [
 ("Three Line Strike",      84, True,  "A", d_three_line_strike, 4,
   "Three falling reds then one green engulfing all three. Bulkowski's highest-ranked reversal."),
 ("Morning Star",           78, True,  "A", d_morning_star,      3,
   "Long red, small star gapping down, long green closing into the first body."),
 ("Three White Soldiers",   75, False, "A", d_three_white,       3,
   "Three rising green candles off a low, each closing near its high."),
 ("Three Outside Up",       74, True,  "A", d_three_outside_up,  3,
   "Bullish engulfing plus a third, higher-closing green candle (self-confirmed)."),
 ("Bullish Abandoned Baby", 70, False, "B", d_abandoned_baby,    3,
   "Island-reversal doji gapping below, then a green gap-up. Rare but strong."),
 ("Three Inside Up",        65, True,  "B", d_three_inside_up,   3,
   "Bullish harami confirmed by a third up-closing candle."),
 ("Bullish Engulfing",      63, True,  "A", d_bull_engulf,       2,
   "Green body fully engulfs the prior red body after a decline."),
 ("Piercing Line",          62, False, "B", d_piercing,          2,
   "Green opens below prior close, then closes above the midpoint of the red body."),
 ("Hammer",                 60, True,  "B", d_hammer,            1,
   "Small body, long lower shadow at a low - lows rejected."),
 ("Tweezer Bottom",         58, False, "B", d_tweezer_bottom,    2,
   "Two candles share the same low - a double rejection."),
 ("Selling Climax",         57, False, "A", d_selling_climax,    1,
   "Wide-range bar on huge volume at the end of a decline - capitulation."),
 ("Stopping Volume / Spring",56,False, "A", d_stopping_volume,   1,
   "New low absorbed on heavy volume, close off the lows."),
 ("Inverted Hammer",        56, False, "C", d_inverted_hammer,   1,
   "Small body, long upper shadow at a low - needs confirmation."),
 ("Bullish Harami",         54, True,  "C", d_bull_harami,       2,
   "Small green inside a long red - early momentum-shift warning."),
 ("No-Supply Bar",          53, False, "A", d_no_supply,         1,
   "Narrow down-bar on dried-up volume - sellers exhausted (VSA)."),
 ("Dragonfly Doji",         52, False, "C", d_dragonfly,         1,
   "Long lower shadow, open=close=high at support."),
]
PATTERNS.sort(key=lambda p:-p[1])
PATTERN_NAMES = [p[0] for p in PATTERNS]

# --------------------------------------------------------- analysis core ----
def analyze_df(df, row, scan_last_n=1):
    """Return list of (pattern_name, match_dict) found on the last scan_last_n bars."""
    out=[]
    if df is None or len(df) < MIN_BARS+5: return out
    A=make_arrays(df); n=A['n']
    for name,wp,src,tier,fn,plen,lit in PATTERNS:
        for back in range(scan_last_n):
            i=n-1-back
            if i<MIN_BARS: break
            try: m=fn(A,i)
            except Exception: m=None
            if m:
                close=A['c'][i]; stop=m['stop']
                risk=(close-stop)/close*100 if close>0 else 0
                vr=A['v'][i]/A['vol_sma'][i] if (ok(A['vol_sma'][i]) and A['vol_sma'][i]>0) else None
                out.append((name, dict(
                    symbol=row.get("symbol"), name=row.get("name",""), exch=row.get("exch",""),
                    sector=row.get("sector",""),
                    date=str(df['date'].iloc[i].date()), bars_ago=back,
                    close=round(close,2), stop=round(stop,2), risk=round(risk,2),
                    target=round(close+2*(close-stop),2),
                    vol_ratio=round(vr,2) if vr is not None else None,
                    vol_confirmed=vol_conf(A,i), note=m['note'])))
                break
    return out

def run_screen(rows, fetch_fn=None, scan_last_n=1, max_workers=8, progress=None, request_delay=0.0):
    """rows: list of dicts (need whatever fetch_fn reads, e.g. security_id or yahoo).
    fetch_fn(row)->df|None. progress: callable(done,total,label). Returns (results,scanned,failed)."""
    fetch_fn = fetch_fn or (lambda row: fetch_ohlcv(row["yahoo"]))
    results={name:[] for name in PATTERN_NAMES}
    scanned=0; failed=[]; total=len(rows); done=0
    def work(row):
        if request_delay: time.sleep(request_delay)
        return row, fetch_fn(row)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs=[ex.submit(work,r) for r in rows]
        for fut in as_completed(futs):
            row,df=fut.result(); done+=1
            if df is None or len(df) < MIN_BARS+5:
                failed.append(row["symbol"])
            else:
                scanned+=1
                for name,m in analyze_df(df,row,scan_last_n):
                    results[name].append(m)
            if progress: progress(done,total,row["symbol"])
    for name in results:
        results[name].sort(key=lambda r:(not r['vol_confirmed'], -(r['vol_ratio'] or 0), r['bars_ago']))
    return results, scanned, failed

# --------------------------------------------------------- HTML report ----
def build_html(results, scanned, failed, scan_last_n=1, title="Bullish Reversal Screener", timeframe="Daily"):
    from datetime import datetime
    unit = "w" if str(timeframe).lower().startswith("w") else "d"
    total=sum(len(v) for v in results.values())
    today=datetime.now().strftime("%d %b %Y %H:%M")
    first_active=next((idx for idx,p in enumerate(PATTERNS) if results[p[0]]), 0)
    tabs=""; panes=""
    for idx,(name,wp,src,tier,fn,plen,lit) in enumerate(PATTERNS):
        rows=results[name]; cnt=len(rows); active="active" if idx==first_active else ""
        srcmark="" if src else " ~est"
        tabs+=(f'<button class="tab {active}" onclick="show({idx})">'
               f'<span class="wp">{wp}%{srcmark}</span><span class="tn">{name}</span>'
               f'<span class="cnt">{cnt}</span></button>')
        if rows:
            trs=""
            def _mc(v):
                if v is None: return "&mdash;"
                return f"{v/1e5:.2f}L Cr" if v>=1e5 else (f"{v:,.0f} Cr" if v>=1000 else f"{v:,.1f} Cr")
            for r in rows:
                badge=('<span class="vok">check vol</span>' if r['vol_confirmed'] else '<span class="vno">-</span>')
                ago="today" if r['bars_ago']==0 else f"{r['bars_ago']}{unit} ago"
                vr=f"{r['vol_ratio']}x" if r['vol_ratio'] is not None else "-"
                trs+=(f"<tr><td class='sym'><a href='{tv_url(r['exch'],r['symbol'])}' "
                      f"target='_blank' rel='noopener'>{r['symbol']}</a><span class='ex'>{r['exch']}</span></td>"
                      f"<td>{r['close']}</td><td>{r['date']}<span class='ago'>{ago}</span></td>"
                      f"<td>{vr} {badge}</td><td>{r['stop']}</td><td>{r['risk']}%</td>"
                      f"<td>{r['target']}</td><td>{_mc(r.get('mcap_cr'))}</td><td class='note'>{r['note']}</td></tr>")
            body=(f"<table><tr><th>Symbol</th><th>Close</th><th>Signal</th><th>Vol vs avg</th>"
                  f"<th>Stop</th><th>Risk</th><th>2R target</th><th>Mkt Cap</th><th>Read</th></tr>{trs}</table>")
        else:
            body="<div class='empty'>No fresh signals in this category.</div>"
        panes+=(f'<div class="pane {active}" id="pane{idx}"><div class="phead"><h2>{name}</h2>'
                f'<span class="pill tier{tier}">Tier {tier}</span>'
                f'<span class="pill wppill">{wp}% hist.{srcmark}</span></div>'
                f'<p class="lit">{lit}</p>'
                f'<p class="gate">Location gate: prior downtrend required'
                f'{" + near a recent low" if plen==1 else ""}. Volume-confirmed first.</p>{body}</div>')
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title><style>
:root{{--ink:#11161d;--mut:#3a4654;--line:#d7dde5;--panel:#f4f7fb;--green:#1b7a2f;--blue:#0d47a1}}
*{{box-sizing:border-box}}body{{margin:0;background:#fff;color:var(--ink);font-size:18px;line-height:1.55;
font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}}
.wrap{{max-width:1180px;margin:0 auto;padding:26px 22px 90px}}h1{{font-size:34px;margin:0 0 4px}}
.sub{{font-size:19px;color:var(--mut);margin:0 0 6px}}
.meta{{font-size:17px;color:var(--mut);background:var(--panel);border:1px solid var(--line);
border-radius:12px;padding:12px 16px;margin:14px 0 20px}}
.tabs{{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:20px}}
.tab{{display:flex;flex-direction:column;gap:1px;cursor:pointer;background:var(--panel);
border:1px solid var(--line);border-radius:12px;padding:9px 13px;font-family:inherit;text-align:left}}
.tab .wp{{font-size:15px;font-weight:800;color:var(--green)}}.tab .tn{{font-size:16px;font-weight:700}}
.tab .cnt{{font-size:14px;color:var(--mut)}}
.tab.active{{border-color:var(--green);border-width:2px;background:#eaf6ec}}
.pane{{display:none}}.pane.active{{display:block}}
.phead{{display:flex;align-items:center;gap:12px;flex-wrap:wrap}}h2{{font-size:28px;margin:6px 0}}
.pill{{font-size:15px;font-weight:800;color:#fff;padding:4px 12px;border-radius:9px}}
.tierA{{background:#1b7a2f}}.tierB{{background:#0d47a1}}.tierC{{background:#8a6d00}}.wppill{{background:#445}}
.lit{{font-size:18px;margin:8px 0 4px}}.gate{{font-size:16px;color:var(--mut);margin:0 0 14px}}
table{{width:100%;border-collapse:collapse;font-size:17px}}
th,td{{padding:10px 12px;border-bottom:1px solid var(--line);text-align:left}}
th{{background:var(--panel);font-size:14px;text-transform:uppercase;letter-spacing:.4px;color:var(--mut)}}
.sym{{font-weight:800;font-size:18px}}.sym a{{color:#0d47a1;text-decoration:none}}
.sym a:hover{{text-decoration:underline}}.ex{{font-size:12px;color:var(--mut);margin-left:6px}}
.ago{{display:block;font-size:13px;color:var(--mut)}}.note{{color:var(--mut);font-size:15px}}
.vok{{color:var(--green);font-weight:800}}.vno{{color:#aaa}}
.empty{{padding:22px;background:var(--panel);border:1px dashed var(--line);border-radius:12px;color:var(--mut)}}
.foot{{margin-top:34px;font-size:15px;color:var(--mut);background:var(--panel);border:1px solid var(--line);
border-radius:12px;padding:16px 18px}}</style></head><body><div class="wrap">
<h1>{title}</h1><p class="sub">Long-only &middot; tabs ordered by historical reversal frequency (highest first)</p>
<div class="meta"><b>{today}</b> &middot; {timeframe} candles &middot; {scanned} stocks scanned &middot; <b>{total}</b> signals &middot;
last {scan_last_n} bar(s). Win-% = historical best-case reversal frequency (a ranking aid, not a tradeable win-rate).</div>
<div class="tabs">{tabs}</div>{panes}
<div class="foot"><b>How to read:</b> strongest setups are volume-confirmed signals in the top tabs at a tested
support level. Stop = pattern low; 2R target assumes entry at last close. Research tool, not investment advice.</div>
<script>function show(i){{document.querySelectorAll('.tab').forEach((t,j)=>t.classList.toggle('active',j===i));
document.querySelectorAll('.pane').forEach((p,j)=>p.classList.toggle('active',j===i));}}</script>
</body></html>"""

# ------------------------------------------------------------- self-test ----
def _df(seq):
    return pd.DataFrame(dict(date=pd.date_range("2024-01-01",periods=len(seq),freq="D"),
        open=[s[0] for s in seq],high=[s[1] for s in seq],low=[s[2] for s in seq],
        close=[s[3] for s in seq],volume=[s[4] for s in seq]))
def _downlead(n=45,start=200.0,step=1.2):
    seq=[];p=start
    for _ in range(n): o=p;c=p-step;seq.append((o,o+0.3,c-0.3,c,100000));p=c
    return seq
def _uplead(n=45,start=120.0,step=1.2):
    seq=[];p=start
    for _ in range(n): o=p;c=p+step;seq.append((o,c+0.3,o-0.3,c,100000));p=c
    return seq
def selftest():
    fails=[]
    def check(name,lead,tail,det,should=True):
        A=make_arrays(_df(lead+tail));i=A['n']-1
        got=det(A,i) is not None
        if got!=should: fails.append(name+("" if should else " (control)"))
        print(f"  {'OK ' if got==should else 'FAIL'} {name}{'' if should else ' [uptrend->reject]'}")
    check("Hammer",_downlead(),[(145.0,146.3,140.0,146.0,120000)],d_hammer)
    check("Hammer",_uplead(),[(145.0,146.3,140.0,146.0,120000)],d_hammer,should=False)
    check("Inverted Hammer",_downlead(),[(145,151,144.7,145.6,120000)],d_inverted_hammer)
    check("Dragonfly Doji",_downlead(),[(145.5,145.6,140,145.5,120000)],d_dragonfly)
    check("Bullish Engulfing",_downlead(),[(146,146.5,143,143.5,90000),(143,149,142.5,148.5,150000)],d_bull_engulf)
    check("Bullish Engulfing",_uplead(),[(146,146.5,143,143.5,90000),(143,149,142.5,148.5,150000)],d_bull_engulf,should=False)
    check("Piercing Line",_downlead(),[(148,148.5,141,141.5,90000),(140,145.6,139.5,145.5,150000)],d_piercing)
    check("Tweezer Bottom",_downlead(),[(146,146.5,142.0,142.5,90000),(142.6,147,142.02,146.8,150000)],d_tweezer_bottom)
    check("Morning Star",_downlead(),[(148,148.5,141,141.5,90000),(140.5,141,139.5,140.0,60000),(140.5,147,140.2,146.5,150000)],d_morning_star)
    check("Three White Soldiers",_downlead(),[(145,147.4,144.8,147.2,120000),(147,149.4,146.8,149.2,130000),(149,151.4,148.8,151.2,140000)],d_three_white)
    check("Three Inside Up",_downlead(),[(149,149.5,141,141.5,90000),(143,146,142.8,145.5,80000),(145.5,150,145,149.5,150000)],d_three_inside_up)
    check("Three Outside Up",_downlead(),[(146,146.5,143.5,144,90000),(143.5,149,143,148.5,150000),(148.5,151,148,150.5,140000)],d_three_outside_up)
    check("Three Line Strike",_downlead(),[(150,150.5,147,147.5,90000),(147.5,148,144,144.5,90000),(144.5,145,141,141.5,90000),(141,151,140.5,150.5,160000)],d_three_line_strike)
    check("Bullish Abandoned Baby",_downlead(),[(148,148.5,143,143.5,90000),(141,141.3,140.7,141.0,60000),(143,147,142.6,146.5,150000)],d_abandoned_baby)
    check("Bullish Harami",_downlead(),[(150,150.5,141,141.5,90000),(143,145,142.8,144.5,70000)],d_bull_harami)
    check("Selling Climax",_downlead(),[(146,146.5,135,136,400000)],d_selling_climax)
    check("Stopping Volume / Spring",_downlead(),[(144,144.5,135,143.0,400000)],d_stopping_volume)
    check("No-Supply Bar",_downlead(),[(145,145.3,144.6,144.8,30000)],d_no_supply)
    print(f"\nSelf-test: {'ALL PASS' if not fails else 'FAILURES: '+', '.join(fails)}")
    return not fails

if __name__=="__main__":
    selftest()
