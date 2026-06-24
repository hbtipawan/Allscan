#!/usr/bin/env python3
"""
marketcap.py — market capitalisation (in INR Crore) for the screener's RESULT stocks.

Primary source is Screener.in (works from cloud datacenter IPs, no key/auth, great NSE
coverage). Yahoo (cookie+crumb) is a fallback for anything Screener misses. Only the
result stocks are looked up (tens, not thousands), and the app caches per day.
Degrades to None on failure (the column shows a dash).
"""
import urllib.request, urllib.parse, re, time, http.cookiejar, json
from concurrent.futures import ThreadPoolExecutor

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
_NUM = re.compile(r'Market Cap.*?<span class="number">\s*([\d,]+)', re.S)

# ------------------------------------------------------------- Screener.in ----
def _screener(symbol):
    url = f"https://www.screener.in/company/{urllib.parse.quote(str(symbol).strip())}/"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "text/html"})
        html = urllib.request.urlopen(req, timeout=12).read().decode("utf-8", "ignore")
    except Exception:
        return None
    m = _NUM.search(html)
    if not m: return None
    try: return float(m.group(1).replace(",", ""))
    except Exception: return None

# ------------------------------------------------------------------- Yahoo ----
def _yahoo_session():
    cj = http.cookiejar.CookieJar()
    op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    op.addheaders = [("User-Agent", UA), ("Accept", "*/*")]
    for u in ("https://fc.yahoo.com", "https://finance.yahoo.com/quote/RELIANCE.NS"):
        try: op.open(u, timeout=10).read()
        except Exception: pass
    crumb = op.open("https://query2.finance.yahoo.com/v1/test/getcrumb", timeout=10).read().decode().strip()
    if not crumb or "<" in crumb: raise RuntimeError("no crumb")
    return op, crumb

def _yahoo_caps(pairs):
    out = {}
    try: op, crumb = _yahoo_session()
    except Exception: return out
    ymap = {f"{s}.{'NS' if e=='NSE' else 'BO'}": (s, e) for s, e in pairs}
    ys = list(ymap.keys())
    for k in range(0, len(ys), 40):
        url = ("https://query2.finance.yahoo.com/v7/finance/quote?symbols="
               + urllib.parse.quote(",".join(ys[k:k+40])) + "&crumb=" + urllib.parse.quote(crumb))
        try:
            d = json.loads(op.open(urllib.request.Request(url), timeout=20).read().decode())
            for r in (d.get("quoteResponse") or {}).get("result") or []:
                mc = r.get("marketCap")
                if mc is None:
                    sh, px = r.get("sharesOutstanding"), r.get("regularMarketPrice")
                    mc = sh*px if (sh and px) else None
                key = ymap.get(r.get("symbol"))
                if key and mc: out[key] = round(mc/1e7, 1)
            time.sleep(0.25)
        except Exception:
            continue
    return out

# --------------------------------------------------------------- public API ----
def get_marketcaps(pairs):
    """pairs: iterable of (symbol, exch in {'NSE','BSE'}). Returns {(symbol,exch): cr|None}."""
    pairs = list(dict.fromkeys((str(s).strip().upper(), e) for s, e in pairs))
    out = {}
    if not pairs: return out
    def work(p):
        v = _screener(p[0]); time.sleep(0.1); return p, v
    try:
        with ThreadPoolExecutor(max_workers=4) as ex:
            for p, v in ex.map(work, pairs): out[p] = v
    except Exception:
        pass
    miss = [p for p in pairs if not out.get(p)]
    if miss:
        try:
            for p, v in _yahoo_caps(miss).items():
                if v: out[p] = v
        except Exception:
            pass
    for p in pairs: out.setdefault(p, None)
    return out

def fmt_cr(v):
    if v is None: return "&mdash;"
    if v >= 1e5: return f"{v/1e5:.2f}L Cr"
    if v >= 1000: return f"{v:,.0f} Cr"
    return f"{v:,.1f} Cr"
