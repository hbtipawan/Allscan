#!/usr/bin/env python3
"""
vcp_backtest.py — point-in-time backtest of VCP breakouts on Upstox daily data.

For each volume-surge breakout to a new local high, we measure the quality of the
base it broke from (tightness, length, contraction, volume dry-up, distance to the
52w high, trend stacking, relative strength) using ONLY bars up to the breakout, then
measure the FORWARD outcome using only later bars (no lookahead). Aggregating across
hundreds of names answers: which screener settings maximise breakout follow-through?
"""
import sys, time, numpy as np, pandas as pd, datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed
import upstox_data as ud

H1, H2 = 20, 40           # forward horizons (trading days)
START_AFTER = 220         # need 200d MA + buffer before first signal
MAX_TIGHT_CAP = 0.08      # capture bases up to 8% wide (so we can study tightness)
MIN_BASE = 3
VOL_SURGE = 1.5           # breakout volume vs 50d avg
NEWHIGH_LB = 15           # breakout = new high over this many bars

def arrays(df):
    c=df['close'].values.astype(float); h=df['high'].values.astype(float)
    l=df['low'].values.astype(float);  v=df['volume'].values.astype(float)
    sma=lambda x,n: pd.Series(x).rolling(n).mean().values
    return dict(c=c,h=h,l=l,v=v,n=len(c),
        ma50=sma(c,50),ma150=sma(c,150),ma200=sma(c,200),vol50=sma(v,50))

def tight_base(A,end,max_tight):
    c=A['c'][end]; hi=A['h'][end]; lo=A['l'][end]; k=1
    if c<=0 or (hi-lo)/c>max_tight: return 0,hi,lo
    for j in range(end-1,max(-1,end-40),-1):
        nhi=max(hi,A['h'][j]); nlo=min(lo,A['l'][j])
        if (nhi-nlo)/c>max_tight: break
        hi,lo=nhi,nlo; k+=1
    return k,hi,lo

def funnel(A,end,c,F=30):
    w=F//3
    rp=lambda a,b:(np.nanmax(A['h'][max(a,0):b+1])-np.nanmin(A['l'][max(a,0):b+1]))/c if b>=max(a,0) else np.nan
    r1=rp(end-F+1,end-2*w); r3=rp(end-w+1,end)
    return (r3/r1) if (r1 and r1>0 and not np.isnan(r1)) else np.nan

def scan_stock(df, nifty_ret_by_date):
    A=arrays(df); n=A['n']; out=[]
    if n < START_AFTER+H2+2: return out
    dts=df['date'].values
    for i in range(START_AFTER, n-H2-1):
        v50=A['vol50'][i]
        if not (v50>0): continue
        surge=A['v'][i]/v50
        if surge < VOL_SURGE: continue
        c=A['c'][i]
        if c<=A['c'][i-1]: continue                                  # up day
        if c < np.nanmax(A['c'][i-NEWHIGH_LB:i]): continue           # new local high (breakout)
        bk,piv,blo = tight_base(A,i-1,MAX_TIGHT_CAP)                 # base ended yesterday
        if bk < MIN_BASE or piv<=0 or c<=piv: continue              # must clear the base pivot
        ma50,ma150,ma200=A['ma50'][i],A['ma150'][i],A['ma200'][i]
        if any(np.isnan(x) for x in (ma50,ma150,ma200)): continue
        hi252=np.nanmax(A['h'][max(0,i-251):i+1]); lo252=np.nanmin(A['l'][max(0,i-251):i+1])
        if c < 1.30*lo252: continue                                  # >=30% above 52w low (baseline)
        nh=(hi252-c)/hi252*100
        seg=A['v'][i-bk:i]; dry=float(np.nanmean(seg))/v50 if v50>0 else np.nan
        contr=funnel(A,i-1,c)
        # relative strength vs nifty (126d)
        rs=np.nan
        if i-126>=0 and A['c'][i-126]>0:
            sret=c/A['c'][i-126]-1
            nd=nifty_ret_by_date.get(pd.Timestamp(dts[i]).normalize(), np.nan)
            rs=(sret-nd)*100 if not np.isnan(nd) else np.nan
        # trend flags
        stacked = c>ma50>ma150>ma200
        ma200_rising = ma200 > A['ma200'][i-20]
        ma50_rising  = ma50  > A['ma50'][i-10]
        strict   = stacked and ma200_rising
        standard = (c>ma50 and c>ma150 and c>ma200 and ma50>ma150 and ma50_rising)
        relaxed  = (c>ma150 and ma50>ma150 and ma50_rising)
        if not relaxed: continue                                     # minimal uptrend floor
        # forward outcome (no lookahead)
        entry=c; stop=blo; risk=entry-stop
        fwd=lambda H: A['c'][i+H]/entry-1
        mfe=lambda H: np.nanmax(A['h'][i+1:i+1+H])/entry-1
        mae=lambda H: np.nanmin(A['l'][i+1:i+1+H])/entry-1
        stopped20 = bool(np.nanmin(A['l'][i+1:i+1+H1]) <= stop)
        r_mult = ((A['c'][i+H1]-entry)/risk) if risk>0 else np.nan
        out.append(dict(
            tight=round((piv-blo)/c*100,2), base_len=int(bk), near_high=round(nh,2),
            contraction=round(contr,3) if not np.isnan(contr) else np.nan,
            dryup=round(dry,3) if not np.isnan(dry) else np.nan,
            rs=round(rs,2) if not np.isnan(rs) else np.nan, surge=round(surge,2),
            strict=strict, standard=standard, relaxed=relaxed,
            ret20=round(fwd(H1)*100,2), ret40=round(fwd(H2)*100,2),
            mfe20=round(mfe(H1)*100,2), mae20=round(mae(H1)*100,2),
            stopped20=stopped20, r20=round(r_mult,3) if not np.isnan(r_mult) else np.nan))
    return out

def main():
    nse=pd.read_csv("EQUITY_L_2.csv").rename(columns=lambda c:c.strip())
    syms=nse["companyId"].astype(str).str.strip().tolist()
    nmap,_=ud.build_symbol_maps()
    universe=[(s,nmap[s]) for s in syms if s in nmap]
    N=int(sys.argv[1]) if len(sys.argv)>1 else 350
    universe=universe[:N]
    to_d=dt.date.today(); from_d=(to_d-dt.timedelta(days=int(4.3*365))).isoformat(); to_s=to_d.isoformat()
    print(f"Backtesting {len(universe)} NSE names, {from_d}..{to_s} (Upstox daily)", file=sys.stderr)

    # Nifty 126d-return-by-date map
    nf=ud.fetch_daily(ud.NIFTY_ROW["instrument_key"], from_d, to_s)
    nret={}
    if nf is not None:
        nc=nf['close'].values
        for k in range(126,len(nf)):
            nret[pd.Timestamp(nf['date'].iloc[k]).normalize()]=nc[k]/nc[k-126]-1

    rows=[]; done=0
    def work(item):
        s,ik=item
        try: return s, ud.fetch_daily(ik, from_d, to_s)
        except Exception: return s, None
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs=[ex.submit(work,it) for it in universe]
        for fut in as_completed(futs):
            s,df=fut.result(); done+=1
            if done%50==0: print(f"  fetched {done}/{len(universe)}", file=sys.stderr)
            if df is None or len(df)<START_AFTER+H2+5: continue
            if np.nanmedian(df['volume'].values) < 50000: continue   # liquidity floor
            for r in scan_stock(df, nret):
                r["symbol"]=s; rows.append(r)
    bt=pd.DataFrame(rows)
    bt.to_csv("vcp_backtest_trades.csv", index=False)
    print(f"\nTotal breakout trades captured: {len(bt)}", file=sys.stderr)
    print("SAVED vcp_backtest_trades.csv", file=sys.stderr)

if __name__=="__main__":
    main()
