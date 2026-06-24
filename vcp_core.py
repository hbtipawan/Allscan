#!/usr/bin/env python3
"""
vcp_core.py — Volatility Contraction Pattern (VCP) engine.
Finds tight, high-grade bases near the highs that are coiling or breaking out
(TARIL-type). Daily or weekly. Imported by the Streamlit app.

Quality logic (Minervini SEPA-inspired):
  GATES: established uptrend (price above rising medium/long MAs, short>medium MA),
         near the recent high (within near_high_pct), >=30% above the 52-period low,
         a TIGHT final base (span <= max_tight, default 5%).
  CLASSIFY: "Coiling" (price sitting just under the pivot) or
            "Breakout" (today clears a prior tight base on a volume surge).
  GRADE A/B/C from: tightness, volatility contraction, volume dry-up,
            proximity to high, base length, relative strength vs Nifty.
"""
import numpy as np, pandas as pd, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import screener_core as sc   # reuse fetch_ohlcv + tv_url

def tf_params(timeframe):
    if timeframe == "Weekly":
        return dict(maf=10, mam=30, mal=40, win=52, mom=26, funnel=15, rng="5y")
    return dict(maf=50, mam=150, mal=200, win=252, mom=126, funnel=30, rng="2y")

def vcp_arrays(df, p):
    c=df['close'].values.astype(float); h=df['high'].values.astype(float)
    l=df['low'].values.astype(float);  v=df['volume'].values.astype(float)
    sma=lambda x,n: pd.Series(x).rolling(n).mean().values
    return dict(c=c,h=h,l=l,v=v, maf=sma(c,p['maf']), mam=sma(c,p['mam']),
        mal=sma(c,p['mal']), vol50=sma(v,50), n=len(c))

def _ok(x): return x is not None and not (isinstance(x,float) and np.isnan(x))

def tight_base(A, end, max_tight):
    """Walk backward from `end` while the running high-low span stays <= max_tight*close.
    Only bars within the tight band are counted. Returns (#bars, pivot_high, base_low)."""
    c=A['c'][end]; hi=A['h'][end]; lo=A['l'][end]; k=1
    if c<=0 or (hi-lo)/c > max_tight: return 0, hi, lo
    for j in range(end-1, max(-1, end-40), -1):
        nhi=max(hi,A['h'][j]); nlo=min(lo,A['l'][j])
        if (nhi-nlo)/c > max_tight: break
        hi,lo=nhi,nlo; k+=1
    return k, hi, lo

def _funnel(A, end, F, c):
    """Range% over three consecutive thirds of the base ending at `end` (old->new)."""
    w=max(F//3,1)
    def rp(a,b):
        a=max(a,0)
        if b<a: return np.nan
        return (np.nanmax(A['h'][a:b+1])-np.nanmin(A['l'][a:b+1]))/c
    return rp(end-F+1,end-2*w), rp(end-2*w+1,end-w), rp(end-w+1,end)

def analyze_vcp(df, row, timeframe="Daily", near_high_pct=0.12, max_tight=0.05,
                min_base=3, strictness="Strict", nifty_mom_ret=0.0):
    p=tf_params(timeframe); A=vcp_arrays(df,p); n=A['n']
    if n < max(p['mal'], p['win'])+5: return None
    i=n-1; c=A['c'][i]
    maf,mam,mal=A['maf'][i],A['mam'][i],A['mal'][i]
    if not all(_ok(x) for x in (c,maf,mam,mal)) or c<=0: return None
    # --- trend template gates (Relaxed / Standard / Strict). Strict = textbook Stage 2 ---
    maf_rising = maf > A['maf'][i-10]
    mal_rising = mal > A['mal'][i-20]
    if strictness=="Strict":
        if not (c>maf>mam>mal and mal_rising): return None
    elif strictness=="Relaxed":
        if not (c>mam and maf>mam and maf_rising): return None
    else:  # Standard
        if not (c>maf and c>mam and c>mal and maf>mam and maf_rising): return None
    hi52=np.nanmax(A['h'][i-p['win']+1:i+1]); lo52=np.nanmin(A['l'][i-p['win']+1:i+1])
    nh=(hi52-c)/hi52
    if nh > near_high_pct: return None                       # within near_high_pct of 52-period high
    if c < 1.30*lo52: return None                            # >=30% above the 52-period low
    vol50=A['vol50'][i]; vol_surge=A['v'][i]/vol50 if _ok(vol50) and vol50>0 else 0
    F=p['funnel']
    def assess(end):
        """Validate a proper VCP base ending at `end`: tight, contracting, quiet."""
        bk,piv,blo=tight_base(A,end,max_tight)
        if bk<min_base or piv<=0: return None
        tight=(piv-blo)/c
        r1,r2,r3=_funnel(A,end,F,c)
        if not _ok(r1) or r1<=0 or not _ok(r3): return None
        contr=r3/r1
        if contr>1.0: return None                           # base must have net-contracted
        seg=A['v'][end-bk+1:end+1]
        dry=float(np.nanmean(seg))/vol50 if _ok(vol50) and vol50>0 else 1.0
        if dry>1.0: return None                             # base must be quiet (volume dry-up)
        return dict(bk=bk,piv=piv,blo=blo,tight=tight,contr=contr,dry=dry)
    status=base=None
    co=assess(i)                                            # coiling: base ends today, price under pivot
    if co and c<=co['piv']*1.001 and c>=co['piv']*(1-0.06):
        status,base="Coiling",co
    else:                                                   # breakout: base ended yesterday, clears pivot on volume
        bo=assess(i-1)
        if bo and c>bo['piv'] and vol_surge>=2.5:           # 2.5x+ surge (backtest: 2.5-4x best)
            status,base="Breakout",bo
    if base is None: return None
    mom=p['mom']; sret=(c/A['c'][i-mom]-1) if i-mom>=0 and A['c'][i-mom]>0 else 0.0
    rs=sret-(nifty_mom_ret or 0.0)
    # weights re-tuned from a 2,072-trade breakout backtest: proximity to 52w high,
    # relative strength, and a 2.5-4x breakout surge were the strongest return drivers.
    s_tight=14*max(0,1-base['tight']/max_tight)
    s_contr=12*float(np.clip(1-base['contr'],0,1))
    s_dry  =10*float(np.clip((1-base['dry'])/0.5,0,1))
    s_near =18*max(0,1-nh/near_high_pct)
    s_base = 8*float(np.clip((base['bk']-min_base)/8,0,1))
    s_rs   =18*float(np.clip(0.5+rs/0.5,0,1))
    s_surge=10*(1.0 if 2.5<=vol_surge<=4.0 else (0.5 if vol_surge>1.5 else 0.0)) if status=="Breakout" \
            else 10*float(np.clip((1-base['dry'])/0.5,0,1))   # coiling: reward quiet base instead
    s_trend=10
    score=round(s_tight+s_contr+s_dry+s_near+s_base+s_rs+s_surge+s_trend)
    grade="A" if score>=75 else "B" if score>=60 else "C"
    return dict(symbol=row['symbol'], name=row.get('name',''), exch=row.get('exch',''),
        sector=row.get('sector',''), close=round(c,2), status=status, grade=grade, score=score,
        base_len=int(base['bk']), tightness=round(base['tight']*100,1),
        contraction=round(base['contr'],2), dryup=round(base['dry'],2),
        near_high=round(nh*100,1), rs=round(rs*100,1), pivot=round(base['piv'],2),
        stop=round(base['blo'],2), dist_pivot=round((base['piv']-c)/c*100,2),
        vol_surge=round(vol_surge,2), target=round(c+2*(c-base['blo']),2),
        date=str(df['date'].iloc[i].date()))

def nifty_mom(timeframe="Daily", fetch_fn=None, nifty_row=None):
    """Benchmark momentum return over the lookback. fetch_fn(row)->df; nifty_row identifies
    the index for the chosen data source. Returns 0.0 if unavailable (RS just neutralises)."""
    p=tf_params(timeframe)
    if fetch_fn is None:
        fetch_fn=lambda row: sc.fetch_ohlcv(row["yahoo"], rng=p['rng'],
                              interval="1wk" if timeframe=="Weekly" else "1d")
    if nifty_row is None:
        nifty_row={"yahoo":"^NSEI","symbol":"NIFTY","exch":"NSE",
                   "security_id":"13","exchange_segment":"IDX_I","instrument":"INDEX"}
    try:
        df=fetch_fn(nifty_row)
    except Exception:
        return 0.0
    if df is None or len(df)<p['mom']+2: return 0.0
    c=df['close'].values
    return float(c[-1]/c[-1-p['mom']]-1)

def build_vcp_html(rows, scanned, failed, timeframe="Daily", summary=""):
    from datetime import datetime
    today=datetime.now().strftime("%d %b %Y %H:%M")
    gcol={"A":"#1b7a2f","B":"#0d47a1","C":"#8a6d00"}
    def _mc(v):
        if v is None: return "&mdash;"
        return f"{v/1e5:.2f}L Cr" if v>=1e5 else (f"{v:,.0f} Cr" if v>=1000 else f"{v:,.1f} Cr")
    trs=""
    for r in rows:
        g=r['grade']; st_="Breakout" if r['status']=="Breakout" else "Coiling"
        stcss="bk" if r['status']=="Breakout" else "co"
        trs+=(f"<tr><td><span class='grade' style='background:{gcol.get(g,'#445')}'>{g}</span></td>"
              f"<td><span class='stt {stcss}'>{st_}</span></td>"
              f"<td class='sym'><a href='{sc.tv_url(r['exch'],r['symbol'])}' target='_blank' "
              f"rel='noopener'>{r['symbol']}</a><span class='ex'>{r['exch']}</span></td>"
              f"<td>{r['close']}</td><td>{r['tightness']}%</td><td>{r['base_len']}</td>"
              f"<td>{r['contraction']}</td><td>{r['dryup']}</td><td>{r['near_high']}%</td>"
              f"<td>{r['rs']}%</td><td>{r['pivot']}</td><td>{r['dist_pivot']}%</td>"
              f"<td>{r['vol_surge']}x</td><td>{r['stop']}</td><td>{r['target']}</td>"
              f"<td>{_mc(r.get('mcap_cr'))}</td><td class='nm'>{r['name']}</td></tr>")
    body=(f"<table><tr><th>Grade</th><th>Status</th><th>Symbol</th><th>Close</th><th>Tight</th>"
          f"<th>Base</th><th>Contr</th><th>Dry</th><th>Near hi</th><th>RS</th><th>Pivot</th>"
          f"<th>To pivot</th><th>Vol</th><th>Stop</th><th>2R tgt</th><th>Mkt Cap</th><th>Company</th></tr>{trs}</table>"
          if rows else "<div class='empty'>No VCP candidates matched.</div>")
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>VCP Scanner</title><style>
:root{{--ink:#11161d;--mut:#3a4654;--line:#d7dde5;--panel:#f4f7fb}}*{{box-sizing:border-box}}
body{{margin:0;background:#fff;color:var(--ink);font-size:17px;font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif}}
.wrap{{max-width:1300px;margin:0 auto;padding:26px 20px 80px}}h1{{font-size:32px;margin:0 0 4px}}
.sub{{font-size:18px;color:var(--mut);margin:0 0 6px}}
.meta{{font-size:16px;color:var(--mut);background:var(--panel);border:1px solid var(--line);
border-radius:12px;padding:12px 16px;margin:14px 0 18px}}
table{{width:100%;border-collapse:collapse;font-size:16px}}
th,td{{padding:9px 9px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}}
th{{background:var(--panel);font-size:13px;text-transform:uppercase;color:var(--mut)}}
.sym{{font-weight:800}}.sym a{{color:#0d47a1;text-decoration:none}}.sym a:hover{{text-decoration:underline}}
.ex{{font-size:11px;color:var(--mut);margin-left:5px}}.nm{{color:var(--mut);font-size:13px;white-space:normal}}
.grade{{color:#fff;font-weight:800;padding:3px 11px;border-radius:8px}}
.stt{{font-weight:800;padding:3px 9px;border-radius:7px;font-size:14px}}
.stt.bk{{background:#e7f6ea;color:#1b7a2f}}.stt.co{{background:#eef2f8;color:#0d47a1}}
.empty{{padding:22px;background:var(--panel);border:1px dashed var(--line);border-radius:12px;color:var(--mut)}}
.foot{{margin-top:28px;font-size:15px;color:var(--mut);background:var(--panel);border:1px solid var(--line);
border-radius:12px;padding:16px 18px}}</style></head><body><div class="wrap">
<h1>VCP Breakout Scanner</h1><p class="sub">Tight bases near the highs &middot; {timeframe} &middot; ranked by grade</p>
<div class="meta"><b>{today}</b> &middot; {scanned} scanned &middot; <b>{len(rows)}</b> candidates &middot; {summary}</div>
{body}
<div class="foot"><b>How to read:</b> <b>Coiling</b> = tight base under the pivot, ready; <b>Breakout</b> = today
cleared the pivot on a volume surge (TARIL-type). Tight = base span %; Dry = base volume vs 50-bar avg
(lower = quieter base); Contr = recent vs earlier range (lower = contracting); enter on a move through the
Pivot, stop at base low. Grade reflects base quality. Research tool, not investment advice.</div>
</div></body></html>"""

def run_vcp_screen(rows, fetch_fn=None, timeframe="Daily", near_high_pct=0.12,
                   max_tight=0.05, min_base=3, strictness="Strict", max_workers=8,
                   progress=None, request_delay=0.0, nifty_ret=0.0):
    fetch_fn = fetch_fn or (lambda row: sc.fetch_ohlcv(row["yahoo"]))
    out=[]; scanned=0; failed=[]; total=len(rows); done=0
    def work(r):
        if request_delay: time.sleep(request_delay)
        return r, fetch_fn(r)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs=[ex.submit(work,r) for r in rows]
        for fut in as_completed(futs):
            r,df=fut.result(); done+=1
            if df is None or len(df)<60: failed.append(r["symbol"])
            else:
                scanned+=1
                try: m=analyze_vcp(df,r,timeframe,near_high_pct,max_tight,min_base,strictness,nifty_ret)
                except Exception: m=None
                if m: out.append(m)
            if progress: progress(done,total,r["symbol"])
    grade_rank={"A":0,"B":1,"C":2}
    out.sort(key=lambda x:(grade_rank.get(x['grade'],9), -x['score']))
    return out, scanned, failed


# ----------------------------------------------------------------- test ----
if __name__=="__main__":
    import sys
    syms=[("TARIL","Transformers & Rectifiers","NSE"),("RELIANCE","Reliance","NSE"),
          ("BHARTIARTL","Bharti Airtel","NSE"),("SIEMENS","Siemens","NSE"),
          ("ABB","ABB","NSE"),("BEL","BEL","NSE"),("HAL","HAL","NSE"),
          ("TITAN","Titan","NSE"),("DIXON","Dixon","NSE"),("POLYCAB","Polycab","NSE"),
          ("APLAPOLLO","APL Apollo","NSE"),("PERSISTENT","Persistent","NSE"),
          ("TRENT","Trent","NSE"),("MAZDOCK","Mazagon","NSE"),("CDSL","CDSL","NSE")]
    tf = "Daily"
    nret = nifty_mom(tf)
    print(f"Nifty {tf} momentum return: {nret*100:.1f}%\n")
    p=tf_params(tf)
    ffn=lambda y: sc.fetch_ohlcv(y, rng=p['rng'], interval="1d")
    for s,nm,ex in syms:
        df=ffn(s+".NS")
        if df is None: print(f"{s}: no data"); continue
        # show trend diagnostics for TARIL specifically
        m=analyze_vcp(df,{"symbol":s,"name":nm,"exch":ex},tf,0.25,0.05,3,"Strict",nret)
        if m:
            print(f"{s:<11} {m['status']:<9} grade {m['grade']} ({m['score']})  "
                  f"tight {m['tightness']}%  base {m['base_len']}  contr {m['contraction']}  "
                  f"dry {m['dryup']}  nearHi {m['near_high']}%  RS {m['rs']}%  pivot {m['pivot']}")
        else:
            A=vcp_arrays(df,p); i=A['n']-1; c=A['c'][i]
            hi=np.nanmax(A['h'][i-p['win']+1:i+1])
            print(f"{s:<11} -- no VCP. c{c:.0f} 50ma{A['maf'][i]:.0f} 150ma{A['mam'][i]:.0f} "
                  f"200ma{A['mal'][i]:.0f} nearHi{(hi-c)/hi*100:.0f}% stacked:"
                  f"{c>A['maf'][i]>A['mam'][i]>A['mal'][i]} 200rising:{A['mal'][i]>A['mal'][i-20]}")
