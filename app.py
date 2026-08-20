
import re
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Momentum Scanner V6.1", page_icon="📈", layout="wide")


DEFAULT_TICKERS = """
AAPL MSFT NVDA AMZN GOOGL META TSLA AVGO AMD NFLX PLTR COIN HOOD RDDT ARM MU
MRNA CRSP EDIT NTLA BEAM RXRX TEM VKTX LLY NVO UNH ISRG
SNDK WDC STX SMCI DELL ORCL CRM SNOW DDOG NET MDB
SOFI UPST AFRM RBLX UBER LYFT SHOP SQ PYPL
SPY QQQ IWM DIA TQQQ SQQQ SOXL SOXS
""".split()

# Large, profitable/liquid proxies used as a "quality/leadership" reference basket.
LEADERS = """
MSFT AAPL NVDA GOOGL AMZN META AVGO ORCL CRM NFLX JPM V MA COST WMT LLY UNH
""".split()

# A deliberately broad high-beta/speculation proxy basket.
# This is NOT a statement that every member is "bad quality"; it is simply a risk-appetite proxy.
SPEC_BASKET = """
PLTR RDDT COIN HOOD MSTR SOFI UPST AFRM RBLX RKLB ASTS IONQ QBTS RGTI QUBT
SOUN BBAI AI PATH TEM RXRX CRSP EDIT NTLA BEAM DNA ACHR JOBY LUNR SMCI CVNA
GME AMC MARA RIOT CLSK HIMS OKLO SMR NNE RIVN LCID QS
""".split()

A_KEYWORDS = [
    "phase 3", "phase iii", "fda approval", "approved by the fda", "nda accepted",
    "breakthrough therapy", "primary endpoint", "acquisition", "acquire", "merger",
    "raises guidance", "raised guidance", "beats and raises", "contract awarded",
    "major contract", "strategic partnership"
]
B_KEYWORDS = [
    "phase 2", "phase ii", "earnings beat", "beats estimates", "upgrade",
    "price target", "partnership", "collaboration", "guidance", "buyback",
    "repurchase", "new product", "launch"
]
NEGATIVE_KEYWORDS = [
    "offering", "public offering", "registered direct", "dilution", "bankruptcy",
    "delisting", "clinical hold", "misses estimates", "cuts guidance", "downgrade"
]

def normalize_tickers(raw):
    vals = re.split(r"[\s,;]+", raw.upper().strip())
    return sorted({x for x in vals if x and re.fullmatch(r"[A-Z0-9.\-^=]+", x)})

def load_sp500():
    tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    return tables[0]["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()

def load_nasdaq100():
    tables = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")
    for t in tables:
        for col in t.columns:
            if str(col).lower() in {"ticker", "symbol"} and len(t) > 50:
                return t[col].astype(str).str.replace(".", "-", regex=False).tolist()
    raise RuntimeError("Could not locate Nasdaq-100 ticker table.")

@st.cache_data(ttl=900, show_spinner=False)
def download_history(tickers):
    return yf.download(
        tickers=tickers,
        period="3mo",
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        threads=True,
        progress=False,
    )

def ticker_frame(blob, ticker, n_tickers):
    if n_tickers == 1:
        df = blob.copy()
    else:
        try:
            df = blob[ticker].copy()
        except Exception:
            return pd.DataFrame()
    return df.dropna(how="all")

def compute_metrics(df):
    if df.empty or "Close" not in df or len(df) < 22:
        return None
    close = df["Close"].dropna()
    volume = df["Volume"].dropna() if "Volume" in df else pd.Series(dtype=float)
    if len(close) < 22:
        return None

    last = float(close.iloc[-1])
    prev = float(close.iloc[-2])
    pct = (last / prev - 1) * 100 if prev else np.nan
    ret5 = (last / float(close.iloc[-6]) - 1) * 100 if len(close) >= 6 else np.nan
    ret20 = (last / float(close.iloc[-21]) - 1) * 100 if len(close) >= 21 else np.nan

    v_last = float(volume.iloc[-1]) if len(volume) else np.nan
    v_base = float(volume.iloc[-21:-1].mean()) if len(volume) >= 21 else np.nan
    rvol = v_last / v_base if v_base and not np.isnan(v_base) else np.nan

    prior20 = float(close.iloc[-21:-1].max())
    prior50 = float(close.iloc[-51:-1].max()) if len(close) >= 51 else np.nan
    prior20low = float(close.iloc[-21:-1].min())
    prior10low = float(close.iloc[-11:-1].min()) if len(close) >= 11 else prior20low
    breakout20 = last > prior20
    breakout50 = bool(last > prior50) if not np.isnan(prior50) else False
    breakdown20 = last < prior20low
    dist20 = (last / prior20 - 1) * 100 if prior20 else np.nan
    dist20low = (last / prior20low - 1) * 100 if prior20low else np.nan
    dist10support = (last / prior10low - 1) * 100 if prior10low else np.nan

    atr_pct = np.nan
    if all(c in df.columns for c in ["High", "Low", "Close"]):
        h = df["High"].astype(float)
        l = df["Low"].astype(float)
        c = df["Close"].astype(float)
        prev_c = c.shift(1)
        tr = pd.concat([(h-l).abs(), (h-prev_c).abs(), (l-prev_c).abs()], axis=1).max(axis=1)
        atr = float(tr.tail(14).mean()) if len(tr.dropna()) >= 5 else np.nan
        atr_pct = atr / last * 100 if last and not np.isnan(atr) else np.nan

    if "High" in df and "Low" in df:
        hi = float(df["High"].dropna().iloc[-1])
        lo = float(df["Low"].dropna().iloc[-1])
        clv = (last - lo) / (hi - lo) if hi > lo else 0.5
    else:
        clv = np.nan

    score = 0
    score += min(max(pct, 0), 25) * 1.4
    if not np.isnan(rvol):
        score += min(max(rvol - 1, 0), 5) * 7
    score += 15 if breakout20 else 0
    score += 5 if breakout50 else 0
    if not np.isnan(clv):
        score += max(min(clv, 1), 0) * 10
    score = min(round(score, 1), 100)

    return {
        "Price": last, "Day %": pct, "5D %": ret5, "20D %": ret20,
        "RVOL": rvol, "Volume": v_last,
        "20D Breakout": breakout20, "50D Breakout": breakout50, "20D Breakdown": breakdown20,
        "vs 20D High %": dist20, "vs 20D Low %": dist20low, "vs 10D Support %": dist10support,
        "ATR %": atr_pct, "Close Location": clv,
        "Momentum Score": score,
    }

SCAN_COLUMNS = [
    "Ticker", "Price", "Day %", "5D %", "20D %", "RVOL", "Volume",
    "20D Breakout", "50D Breakout", "20D Breakdown", "vs 20D High %",
    "vs 20D Low %", "vs 10D Support %", "ATR %", "Close Location",
    "Momentum Score",
]

def scan_universe(tickers, progress=None):
    rows = []
    chunk_size = 120
    for start in range(0, len(tickers), chunk_size):
        chunk = tickers[start:start + chunk_size]
        try:
            blob = download_history(chunk)
            for t in chunk:
                m = compute_metrics(ticker_frame(blob, t, len(chunk)))
                if m:
                    rows.append({"Ticker": t, **m})
        except Exception:
            for t in chunk:
                try:
                    blob = download_history([t])
                    m = compute_metrics(ticker_frame(blob, t, 1))
                    if m:
                        rows.append({"Ticker": t, **m})
                except Exception:
                    pass
        if progress:
            progress.progress(min((start + len(chunk)) / len(tickers), 1.0))
    # Always return a stable schema. This prevents KeyError when Yahoo/yfinance
    # temporarily returns no usable rows for an otherwise valid ticker.
    return pd.DataFrame(rows, columns=SCAN_COLUMNS)

@st.cache_data(ttl=900, show_spinner=False)
def get_news(ticker):
    items = []
    try:
        raw = yf.Ticker(ticker).news or []
        for x in raw[:8]:
            c = x.get("content", x)
            title = c.get("title") or x.get("title") or ""
            provider = c.get("provider", {})
            publisher = provider.get("displayName") if isinstance(provider, dict) else ""
            canonical = c.get("canonicalUrl", {})
            url = canonical.get("url") if isinstance(canonical, dict) else ""
            items.append({"title": title, "publisher": publisher or "", "url": url or ""})
    except Exception:
        pass
    return items

def grade_news(items):
    joined = " ".join(i.get("title","").lower() for i in items)
    if any(k in joined for k in NEGATIVE_KEYWORDS):
        return "Risk", "Headline set contains a dilution / downgrade / other negative-risk keyword."
    if any(k in joined for k in A_KEYWORDS):
        return "A", "Strong catalyst keyword found."
    if any(k in joined for k in B_KEYWORDS):
        return "B", "Moderate catalyst keyword found."
    if items:
        return "C", "News exists, but no strong catalyst keyword was detected automatically."
    return "?", "No usable headline returned."

def clamp(x, lo=0, hi=100):
    return max(lo, min(hi, x))

def froth_score(spec, leaders):
    """Transparent heuristic score, not a market-crash predictor."""
    if spec.empty or leaders.empty:
        return None

    smed = spec["Day %"].median()
    lmed = leaders["Day %"].median()
    spread = smed - lmed

    s5 = spec["5D %"].median()
    l5 = leaders["5D %"].median()
    spread5 = s5 - l5

    pct_up10 = (spec["Day %"] >= 10).mean() * 100
    pct_rvol2 = (spec["RVOL"].fillna(0) >= 2).mean() * 100
    pct_break = spec["20D Breakout"].mean() * 100
    leader_break = leaders["20D Breakout"].mean() * 100

    # Five components, each intentionally capped.
    c1 = clamp((spread + 1.0) * 8, 0, 25)
    c2 = clamp((spread5 + 2.0) * 2.5, 0, 20)
    c3 = clamp(pct_up10 * 1.2, 0, 20)
    c4 = clamp(pct_rvol2 * 0.55, 0, 20)
    c5 = clamp((pct_break - leader_break + 10) * 0.75, 0, 15)
    total = round(clamp(c1 + c2 + c3 + c4 + c5), 1)

    if total < 30:
        label = "Normal"
        action = "Speculation is not broadly dominant."
    elif total < 50:
        label = "Warm"
        action = "Risk appetite is elevated; normal sizing, but avoid weak catalysts."
    elif total < 70:
        label = "Hot"
        action = "Speculative breadth is broadening; reduce chase distance and tighten entries."
    elif total < 85:
        label = "Frothy"
        action = "Low-quality/high-beta leadership is unusually strong; consider smaller position sizes."
    else:
        label = "Extreme"
        action = "Very speculative tape. Treat gap-chasing and leveraged products as high-risk."

    return {
        "score": total, "label": label, "action": action,
        "Spec median Day %": smed, "Leader median Day %": lmed,
        "Day spread": spread, "Spec median 5D %": s5, "Leader median 5D %": l5,
        "5D spread": spread5, "Spec >10% today": pct_up10,
        "Spec RVOL >=2": pct_rvol2, "Spec 20D breakouts": pct_break,
        "Leader 20D breakouts": leader_break,
    }



# -----------------------------
# Buy Score / leveraged-product engine
# -----------------------------
# Direction: +1 = leveraged long; -1 = inverse/short product.
# For sector/index products, "underlying" is a liquid proxy used for signal quality,
# not a claim that the ETF legally tracks that ETF itself.
LEVERAGED_PRODUCTS = {
    # Single-stock leverage
    "MRNX": {"underlying":"MRNA", "leverage":2, "direction":1, "proxy":False},
    "NVDL": {"underlying":"NVDA", "leverage":2, "direction":1, "proxy":False},
    "NVDU": {"underlying":"NVDA", "leverage":2, "direction":1, "proxy":False},
    "NVDD": {"underlying":"NVDA", "leverage":2, "direction":-1, "proxy":False},
    "TSLL": {"underlying":"TSLA", "leverage":2, "direction":1, "proxy":False},
    "TSLQ": {"underlying":"TSLA", "leverage":2, "direction":-1, "proxy":False},
    "AMDL": {"underlying":"AMD", "leverage":2, "direction":1, "proxy":False},
    "AMDS": {"underlying":"AMD", "leverage":1, "direction":-1, "proxy":False},
    "GGLL": {"underlying":"GOOGL", "leverage":2, "direction":1, "proxy":False},
    "GGLS": {"underlying":"GOOGL", "leverage":1, "direction":-1, "proxy":False},
    "AAPU": {"underlying":"AAPL", "leverage":2, "direction":1, "proxy":False},
    "AAPD": {"underlying":"AAPL", "leverage":1, "direction":-1, "proxy":False},
    "MSFU": {"underlying":"MSFT", "leverage":2, "direction":1, "proxy":False},
    "MSFD": {"underlying":"MSFT", "leverage":1, "direction":-1, "proxy":False},
    "AMZU": {"underlying":"AMZN", "leverage":2, "direction":1, "proxy":False},
    "AMZD": {"underlying":"AMZN", "leverage":1, "direction":-1, "proxy":False},
    "CONL": {"underlying":"COIN", "leverage":2, "direction":1, "proxy":False},
    "MSTU": {"underlying":"MSTR", "leverage":2, "direction":1, "proxy":False},
    "MSTX": {"underlying":"MSTR", "leverage":2, "direction":1, "proxy":False},
    "MSTZ": {"underlying":"MSTR", "leverage":2, "direction":-1, "proxy":False},

    # Index / sector leverage: underlying field is the signal proxy.
    "TQQQ": {"underlying":"QQQ", "leverage":3, "direction":1, "proxy":True},
    "SQQQ": {"underlying":"QQQ", "leverage":3, "direction":-1, "proxy":True},
    "QLD":  {"underlying":"QQQ", "leverage":2, "direction":1, "proxy":True},
    "QID":  {"underlying":"QQQ", "leverage":2, "direction":-1, "proxy":True},
    "UPRO": {"underlying":"SPY", "leverage":3, "direction":1, "proxy":True},
    "SPXU": {"underlying":"SPY", "leverage":3, "direction":-1, "proxy":True},
    "SOXL": {"underlying":"SMH", "leverage":3, "direction":1, "proxy":True},
    "SOXS": {"underlying":"SMH", "leverage":3, "direction":-1, "proxy":True},
    "TECL": {"underlying":"XLK", "leverage":3, "direction":1, "proxy":True},
    "TECS": {"underlying":"XLK", "leverage":3, "direction":-1, "proxy":True},
    "LABU": {"underlying":"XBI", "leverage":3, "direction":1, "proxy":True},
    "LABD": {"underlying":"XBI", "leverage":3, "direction":-1, "proxy":True},
    "FAS":  {"underlying":"XLF", "leverage":3, "direction":1, "proxy":True},
    "FAZ":  {"underlying":"XLF", "leverage":3, "direction":-1, "proxy":True},
    "TNA":  {"underlying":"IWM", "leverage":3, "direction":1, "proxy":True},
    "TZA":  {"underlying":"IWM", "leverage":3, "direction":-1, "proxy":True},
    "GUSH": {"underlying":"XOP", "leverage":2, "direction":1, "proxy":True},
    "DRIP": {"underlying":"XOP", "leverage":2, "direction":-1, "proxy":True},
    "ERX":  {"underlying":"XLE", "leverage":2, "direction":1, "proxy":True},
    "ERY":  {"underlying":"XLE", "leverage":2, "direction":-1, "proxy":True},
    "NAIL": {"underlying":"XHB", "leverage":3, "direction":1, "proxy":True},
    "NUGT": {"underlying":"GDX", "leverage":2, "direction":1, "proxy":True},
    "DUST": {"underlying":"GDX", "leverage":2, "direction":-1, "proxy":True},
    "JNUG": {"underlying":"GDXJ", "leverage":2, "direction":1, "proxy":True},
    "JDST": {"underlying":"GDXJ", "leverage":2, "direction":-1, "proxy":True},
    "YINN": {"underlying":"FXI", "leverage":3, "direction":1, "proxy":True},
    "YANG": {"underlying":"FXI", "leverage":3, "direction":-1, "proxy":True},
}

# Optional sector hints for the Market/Sector component. Unknown tickers receive a neutral sector score.
TICKER_GROUP_HINTS = {
    **{x:"Semiconductors" for x in "NVDA AMD AVGO MU ARM SNDK WDC STX SMCI DELL INTC QCOM TSM ASML SMH SOXX XSD".split()},
    **{x:"Software / Cloud" for x in "PLTR CRM SNOW DDOG NET MDB ORCL MSFT IGV SKYY CLOU".split()},
    **{x:"Communication / Internet" for x in "META GOOGL GOOG NFLX RDDT".split()},
    **{x:"Consumer Discretionary" for x in "TSLA AMZN RBLX UBER LYFT".split()},
    **{x:"Health Care" for x in "LLY NVO UNH ISRG TEM".split()},
    **{x:"Biotech" for x in "MRNA CRSP EDIT NTLA BEAM RXRX VKTX XBI IBB".split()},
    **{x:"Financials / Banks" for x in "JPM BAC C GS V MA SOFI HOOD XLF KBE KRE".split()},
    **{x:"Cybersecurity" for x in "CRWD PANW ZS FTNT CIBR HACK BUG".split()},
    **{x:"Energy" for x in "XLE XOP OIH".split()},
    **{x:"Homebuilders" for x in "XHB ITB".split()},
    **{x:"Growth" for x in "QQQ SPYG IWF".split()},
    **{x:"Small Caps" for x in "IWM IJR".split()},
}

CATALYST_POINTS = {"A":20.0, "B":15.0, "C":8.0, "?":4.0, "Risk":0.0}


def _trend_component(m, direction=1):
    r5 = float(m.get("5D %", 0) or 0) * direction
    r20 = float(m.get("20D %", 0) or 0) * direction
    if r5 >= 10: s5 = 8
    elif r5 >= 5: s5 = 7
    elif r5 >= 2: s5 = 6
    elif r5 >= 0: s5 = 5
    elif r5 >= -3: s5 = 3
    else: s5 = 1
    if r20 >= 20: s20 = 8
    elif r20 >= 10: s20 = 7
    elif r20 >= 5: s20 = 6
    elif r20 >= 0: s20 = 5
    elif r20 >= -5: s20 = 3
    else: s20 = 1
    confirm = 4 if (bool(m.get("20D Breakout")) if direction == 1 else bool(m.get("20D Breakdown"))) else 0
    return float(clamp(s5 + s20 + confirm, 0, 20))


def _entry_component(m, direction=1):
    # 0-20 raw points. Strong momentum can still have poor entry quality when too extended.
    day = float(m.get("Day %", 0) or 0) * direction
    dist = float(m.get("vs 20D High %", 0) or 0) if direction == 1 else -float(m.get("vs 20D Low %", 0) or 0)
    clv = m.get("Close Location", np.nan)
    atrp = m.get("ATR %", np.nan)
    s = 20.0
    if day > 25: s -= 13
    elif day > 15: s -= 10
    elif day > 10: s -= 6
    elif day > 6: s -= 3
    elif day < -4: s -= 3

    if dist > 15: s -= 8
    elif dist > 8: s -= 5
    elif dist > 4: s -= 2
    elif -3 <= dist <= 2: s += 1

    if not np.isnan(clv):
        directional_close = clv if direction == 1 else (1 - clv)
        if directional_close >= 0.70: s += 2
        elif directional_close <= 0.25: s -= 2
    if not np.isnan(atrp):
        if atrp > 10: s -= 4
        elif atrp > 7: s -= 2
        elif atrp > 5: s -= 1
    return float(clamp(s, 0, 20))


def _volume_component(m):
    r = m.get("RVOL", np.nan)
    if np.isnan(r): return 2.0
    if r >= 5: return 15.0
    if r >= 4: return 14.0
    if r >= 3: return 12.0
    if r >= 2: return 10.0
    if r >= 1.5: return 8.0
    if r >= 1.2: return 6.0
    if r >= 1.0: return 4.0
    return 2.0


def _market_component(regime, direction=1):
    name = regime.get("regime", "Neutral / Mixed") if regime else "Neutral / Mixed"
    long_map = {
        "Healthy Risk-On":10, "Risk-On":9, "Compression / Watch Breakouts":7,
        "Neutral / Mixed":6, "Conflicted Rally":5, "Orderly Weakness":4,
        "Risk-Off Lean":2, "Risk-Off":0,
    }
    if direction == 1:
        return float(long_map.get(name, 5))
    # Inverse products benefit from weak tape, but still get penalized for chaotic conflict.
    inv_map = {
        "Risk-Off":10, "Risk-Off Lean":9, "Orderly Weakness":7,
        "Neutral / Mixed":5, "Conflicted Rally":5, "Compression / Watch Breakouts":4,
        "Risk-On":2, "Healthy Risk-On":0,
    }
    return float(inv_map.get(name, 5))


def _sector_component(ticker, rotation, direction=1):
    group = TICKER_GROUP_HINTS.get(str(ticker).upper())
    if group is None or rotation is None or rotation.empty:
        return 2.5, group
    hit = rotation[rotation["Group"] == group]
    if hit.empty:
        return 2.5, group
    leadership = float(hit.iloc[0]["Leadership Score"])
    s = leadership / 20.0  # 0..5
    if direction == -1:
        s = 5.0 - s
    return float(clamp(s, 0, 5)), group


def _risk_reward_component(m, direction=1):
    atrp = m.get("ATR %", np.nan)
    if np.isnan(atrp): vol_pts = 2.0
    elif atrp <= 3: vol_pts = 5.0
    elif atrp <= 5: vol_pts = 4.0
    elif atrp <= 7: vol_pts = 3.0
    elif atrp <= 10: vol_pts = 2.0
    else: vol_pts = 1.0

    if direction == 1:
        d = m.get("vs 10D Support %", np.nan)
    else:
        # For shorts, distance above the prior 20D low approximates how far price could snap back.
        d = abs(float(m.get("vs 20D Low %", np.nan))) if not np.isnan(m.get("vs 20D Low %", np.nan)) else np.nan
    if np.isnan(d): support_pts = 2.0
    elif d <= 5: support_pts = 5.0
    elif d <= 10: support_pts = 4.0
    elif d <= 18: support_pts = 3.0
    elif d <= 30: support_pts = 1.0
    else: support_pts = 0.0
    return float(clamp(vol_pts + support_pts, 0, 10))


def buy_score(m, catalyst_grade="?", regime=None, rotation=None, ticker="", direction=1):
    trend = _trend_component(m, direction)
    catalyst = CATALYST_POINTS.get(catalyst_grade, 4.0)
    entry_raw = _entry_component(m, direction)  # max 20
    volume = _volume_component(m)
    market = _market_component(regime, direction)
    sector, group = _sector_component(ticker, rotation, direction)
    rr = _risk_reward_component(m, direction)
    total = round(clamp(trend + catalyst + entry_raw + volume + market + sector + rr, 0, 100), 1)
    entry_quality = round(entry_raw * 5, 1)
    if total >= 85: label = "Strong Buy Setup" if direction == 1 else "Strong Bear Setup"
    elif total >= 75: label = "Buy / Tactical" if direction == 1 else "Short / Tactical"
    elif total >= 65: label = "Watch"
    elif total >= 50: label = "Weak Setup"
    else: label = "Avoid"
    return {
        "score": total, "label": label, "entry_quality": entry_quality, "group": group,
        "components": {
            "Trend / Relative Strength": trend,
            "Catalyst": catalyst,
            "Entry Quality": entry_raw,
            "RVOL / Volume": volume,
            "Market Regime": market,
            "Sector Regime": sector,
            "Risk / Reward": rr,
        }
    }


def leverage_verdict(score_result, leverage=2, product_day=np.nan):
    score = score_result["score"]
    entry = score_result["entry_quality"]
    # Chasing a leveraged product after an already extreme one-day move overrides a good underlying score.
    extreme_product = False
    if not np.isnan(product_day):
        extreme_product = abs(product_day) >= (28 if leverage >= 3 else 20)
    if entry < 45 or extreme_product:
        return "DO NOT CHASE"
    hurdle = 3 if leverage >= 3 else 0
    if score >= 88 + hurdle and entry >= 72:
        return "TACTICAL GO"
    if score >= 78 + hurdle and entry >= 60:
        return "TACTICAL"
    if score >= 68:
        return "WATCH / WAIT FOR ENTRY"
    return "AVOID"


def product_info(ticker, override_underlying=""):
    t = ticker.upper().strip()
    if t in LEVERAGED_PRODUCTS:
        info = dict(LEVERAGED_PRODUCTS[t])
        if override_underlying.strip():
            info["underlying"] = override_underlying.upper().strip()
            info["proxy"] = True
        info["product"] = t
        info["is_leveraged"] = True
        return info
    if override_underlying.strip():
        return {"product":t, "underlying":override_underlying.upper().strip(), "leverage":2,
                "direction":1, "proxy":True, "is_leveraged":True}
    return {"product":t, "underlying":t, "leverage":1, "direction":1, "proxy":False, "is_leveraged":False}


def enrich_buy_scores(out, regime=None, rotation=None):
    if out.empty:
        return out
    result = out.copy()
    cols = ["Catalyst Grade", "Buy Score", "Entry Quality", "Setup", "Score Basis", "Leverage Verdict"]
    for c in cols:
        result[c] = "" if c in ["Catalyst Grade","Setup","Score Basis","Leverage Verdict"] else np.nan

    for idx, row in result.iterrows():
        ticker = row["Ticker"]
        info = product_info(ticker)
        basis_ticker = info["underlying"]
        basis_metrics = row
        product_day = float(row.get("Day %", np.nan))
        if info["is_leveraged"]:
            udf = scan_universe([basis_ticker])
            if not udf.empty:
                basis_metrics = udf.iloc[0]
            else:
                result.at[idx, "Score Basis"] = basis_ticker
                result.at[idx, "Setup"] = "No underlying data"
                continue
        news = get_news(basis_ticker)
        grade, _ = grade_news(news)
        score = buy_score(basis_metrics, grade, regime, rotation, basis_ticker, info["direction"])
        result.at[idx, "Catalyst Grade"] = grade
        result.at[idx, "Buy Score"] = score["score"]
        result.at[idx, "Entry Quality"] = score["entry_quality"]
        result.at[idx, "Setup"] = score["label"]
        result.at[idx, "Score Basis"] = basis_ticker if info["is_leveraged"] else ticker
        if info["is_leveraged"]:
            result.at[idx, "Leverage Verdict"] = leverage_verdict(score, info["leverage"], product_day)
    return result

def market_regime():
    """Classify SPY/QQQ + VIX combinations using daily data."""
    tickers = ["SPY", "QQQ", "^VIX"]
    df = scan_universe(tickers)
    if df.empty:
        return None
    d = {r["Ticker"]: r for _, r in df.iterrows()}
    if not all(x in d for x in tickers):
        return None

    spy, qqq, vix = d["SPY"], d["QQQ"], d["^VIX"]
    equity = np.nanmean([spy["Day %"], qqq["Day %"]])
    vix_day = vix["Day %"]
    vix_level = vix["Price"]

    # Small dead-zone prevents tiny moves from being overinterpreted.
    eq_up = equity > 0.25
    eq_down = equity < -0.25
    vix_up = vix_day > 2.0
    vix_down = vix_day < -2.0

    if eq_up and vix_down:
        regime = "Healthy Risk-On"
        meaning = "SPY/QQQ are rising while VIX is falling: the cleanest environment here for ordinary long momentum."
        risk_adj = -4
    elif eq_up and vix_up:
        regime = "Conflicted Rally"
        meaning = "Stocks are rising but VIX is also rising: upside is occurring alongside demand for protection. Avoid assuming the rally is low-risk."
        risk_adj = 5
    elif eq_down and vix_up:
        regime = "Risk-Off"
        meaning = "SPY/QQQ are falling while VIX is rising: unfavorable backdrop for aggressive high-beta long momentum."
        risk_adj = 10
    elif abs(equity) <= 0.25 and vix_down:
        regime = "Compression / Watch Breakouts"
        meaning = "Indexes are roughly flat while VIX falls. Conditions are calming, but price still needs to confirm direction."
        risk_adj = -1
    elif eq_down and vix_down:
        regime = "Orderly Weakness"
        meaning = "Stocks are slipping without a volatility shock. Weak tape, but not classic panic."
        risk_adj = 2
    elif eq_up:
        regime = "Risk-On"
        meaning = "Indexes are rising and VIX is not sending a strong contrary signal."
        risk_adj = -2
    elif eq_down:
        regime = "Risk-Off Lean"
        meaning = "Indexes are falling; keep momentum entries selective."
        risk_adj = 4
    else:
        regime = "Neutral / Mixed"
        meaning = "Neither index direction nor volatility is giving a strong combined signal."
        risk_adj = 0

    # Very low VIX can be complacency rather than automatically bullish.
    complacency = vix_level < 13
    if complacency:
        meaning += " VIX is also below 13, so very low implied volatility may represent complacency if speculative breadth is simultaneously extreme."

    return {
        "regime": regime,
        "meaning": meaning,
        "equity_day": equity,
        "spy_day": spy["Day %"],
        "qqq_day": qqq["Day %"],
        "vix_level": vix_level,
        "vix_day": vix_day,
        "vix_5d": vix["5D %"],
        "risk_adjustment": risk_adj,
        "complacency": complacency,
    }



# Broad sector + industry + factor leadership universe.
# Multiple related proxies are intentionally included so a single ETF/index does not dominate the signal.
ROTATION_GROUPS = {
    "Semiconductors": ["SMH", "SOXX", "XSD"],
    "Software / Cloud": ["IGV", "SKYY", "CLOU"],
    "Cybersecurity": ["CIBR", "HACK", "BUG"],
    "Communication / Internet": ["XLC", "FDN"],
    "Consumer Discretionary": ["XLY", "RTH"],
    "Consumer Staples": ["XLP"],
    "Financials / Banks": ["XLF", "KBE", "KRE"],
    "Industrials": ["XLI", "PAVE"],
    "Aerospace / Defense": ["ITA", "XAR"],
    "Energy": ["XLE", "XOP", "OIH"],
    "Utilities": ["XLU"],
    "Health Care": ["XLV", "IHI"],
    "Biotech": ["XBI", "IBB"],
    "Materials / Metals": ["XLB", "XME"],
    "Real Estate": ["XLRE", "IYR"],
    "Homebuilders": ["XHB", "ITB"],
    "Transportation": ["IYT"],
    "Small Caps": ["IWM", "IJR"],
    "Mid Caps": ["MDY"],
    "Growth": ["SPYG", "IWF"],
    "Value": ["SPYV", "IWD"],
}

def rotation_dashboard():
    tickers = sorted(set(["SPY", "QQQ"] + [t for vals in ROTATION_GROUPS.values() for t in vals]))
    raw = scan_universe(tickers)
    if raw.empty:
        return pd.DataFrame(), raw

    bench = raw[raw["Ticker"] == "SPY"]
    if bench.empty:
        return pd.DataFrame(), raw
    spy = bench.iloc[0]

    rows = []
    for group, proxies in ROTATION_GROUPS.items():
        g = raw[raw["Ticker"].isin(proxies)].copy()
        if g.empty:
            continue

        day = g["Day %"].median()
        d5 = g["5D %"].median()
        d20 = g["20D %"].median()
        r1 = day - spy["Day %"]
        r5 = d5 - spy["5D %"]
        r20 = d20 - spy["20D %"]
        breakout_breadth = g["20D Breakout"].mean() * 100
        avg_rvol = g["RVOL"].replace([np.inf, -np.inf], np.nan).median()

        # Relative-strength score emphasizes persistence over one-day noise.
        rs_score = clamp(50 + r1 * 4 + r5 * 2.2 + r20 * 1.1 + (breakout_breadth - 50) * 0.12, 0, 100)

        if r1 > 0 and r5 > 0 and r20 > 0:
            state = "Leading"
        elif r5 > 0 and r20 > 0 and r1 < 0:
            state = "Pullback in Leader"
        elif r1 > 0 and r5 > 0 and r20 <= 0:
            state = "Emerging / Improving"
        elif r1 < 0 and r5 < 0 and r20 < 0:
            state = "Losing Leadership"
        elif r1 < 0 and r5 < 0 and r20 >= 0:
            state = "Deteriorating"
        else:
            state = "Mixed / Transition"

        rows.append({
            "Group": group,
            "Proxies": ", ".join(g["Ticker"].tolist()),
            "Day %": day,
            "5D %": d5,
            "20D %": d20,
            "RS vs SPY 1D": r1,
            "RS vs SPY 5D": r5,
            "RS vs SPY 20D": r20,
            "20D Breakout Breadth %": breakout_breadth,
            "Median RVOL": avg_rvol,
            "Leadership Score": round(rs_score, 1),
            "State": state,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["Leadership Score", "RS vs SPY 5D"], ascending=False)
    return out, raw



# -----------------------------
# V5 bilingual UI + daily workflow
# -----------------------------
LANG = st.sidebar.radio("语言 / Language", ["中文", "English"], horizontal=True)
ZH = LANG == "中文"
def tr(zh, en):
    return zh if ZH else en


GROUP_ZH = {
    "Semiconductors":"半导体", "Software / Cloud":"软件 / 云", "Cybersecurity":"网络安全",
    "Communication / Internet":"通信 / 互联网", "Consumer Discretionary":"可选消费",
    "Consumer Staples":"必需消费", "Financials / Banks":"金融 / 银行", "Industrials":"工业",
    "Aerospace / Defense":"航空航天 / 国防", "Energy":"能源", "Utilities":"公用事业",
    "Health Care":"医疗保健", "Biotech":"生物科技", "Materials / Metals":"材料 / 金属",
    "Real Estate":"房地产", "Homebuilders":"房屋建筑", "Transportation":"交通运输",
    "Small Caps":"小盘股", "Mid Caps":"中盘股", "Growth":"成长", "Value":"价值"
}
STATE_ZH = {
    "Leading":"领先", "Pullback in Leader":"强势板块回调", "Emerging / Improving":"转强 / 改善中",
    "Losing Leadership":"失去领导力", "Deteriorating":"恶化中", "Mixed / Transition":"混合 / 过渡"
}
def disp_group(x):
    return GROUP_ZH.get(x, x) if ZH else x

def disp_state(x):
    return STATE_ZH.get(x, x) if ZH else x

def localized_rotation(df):
    if not ZH or df.empty:
        return df
    out=df.copy()
    out["Group"] = out["Group"].map(disp_group)
    out["State"] = out["State"].map(disp_state)
    return out.rename(columns={
        "Group":"梯队", "Proxies":"代理 ETF", "Day %":"当日 %", "5D %":"5日 %", "20D %":"20日 %",
        "RS vs SPY 1D":"相对 SPY 1日", "RS vs SPY 5D":"相对 SPY 5日", "RS vs SPY 20D":"相对 SPY 20日",
        "20D Breakout Breadth %":"20日突破广度 %", "Median RVOL":"RVOL 中位数",
        "Leadership Score":"领导力分数", "State":"状态"
    })

def localized_momentum(df):
    if not ZH or df.empty:
        return df
    return df.rename(columns={
        "Ticker":"代码", "Price":"价格", "Day %":"当日 %", "5D %":"5日 %", "20D %":"20日 %",
        "RVOL":"相对成交量", "Volume":"成交量", "20D Breakout":"突破20日高点",
        "50D Breakout":"突破50日高点", "vs 20D High %":"距20日高点 %",
        "Close Location":"收盘位置", "Momentum Score":"动量分数", "20D Breakdown":"跌破20日低点",
        "vs 20D Low %":"距20日低点 %", "vs 10D Support %":"距10日支撑 %", "ATR %":"ATR %",
        "Catalyst Grade":"催化剂等级", "Buy Score":"Buy Score", "Entry Quality":"入场质量",
        "Setup":"Setup", "Score Basis":"评分依据", "Leverage Verdict":"杠杆结论"
    })

st.title("Momentum + Froth + Rotation Scanner V6.1")
st.caption(tr(
    "催化剂动量 + 市场过热 + VIX 市场环境 + 板块轮动。研究工具，不是券商级实时行情。",
    "Catalyst momentum + market froth + VIX regime + sector rotation. Research tool only; not broker-grade real-time data."
))

TAB_NAMES = [
    tr("每日流程", "Daily Workflow"),
    tr("动量扫描", "Momentum Scanner"),
    tr("Buy Score", "Buy Score"),
    tr("市场过热", "Market Froth"),
    tr("VIX + 市场环境", "VIX + Market Regime"),
    tr("板块轮动", "Sector Rotation"),
]
tab0, tab1, tabbuy, tab2, tab3, tab4 = st.tabs(TAB_NAMES)

# Shared momentum settings
with st.sidebar:
    st.header(tr("动量设置", "Momentum Settings"))
    opts = {
        tr("核心观察池", "Core watchlist"): "Core watchlist",
        tr("自定义股票", "Custom tickers"): "Custom tickers",
        "S&P 500": "S&P 500",
        "Nasdaq-100": "Nasdaq-100",
        "S&P 500 + Nasdaq-100": "S&P 500 + Nasdaq-100",
    }
    label = st.selectbox(tr("股票范围", "Universe"), list(opts.keys()))
    universe_mode = opts[label]
    custom = st.text_area(tr("自定义 Ticker", "Custom tickers"), value="MRNA MRNX NVDA AMD SNDK PLTR")
    min_price = st.number_input(tr("最低股价", "Min price"), 0.0, 10000.0, 5.0, 1.0)
    min_gain = st.number_input(tr("最低当日涨幅 %", "Min daily gain %"), -100.0, 500.0, 5.0, 1.0)
    min_rvol = st.number_input(tr("最低 RVOL", "Min RVOL"), 0.0, 50.0, 1.5, 0.1)
    min_volume = st.number_input(tr("最低成交量", "Min volume"), 0, 1000000000, 500000, 100000)
    require_breakout = st.checkbox(tr("必须突破 20 日高点", "Require 20-day breakout"), value=False)
    top_n = st.slider(tr("显示候选数量", "Show top candidates"), 5, 50, 15)
    include_buy_score = st.checkbox(tr("在候选表计算 Buy Score", "Calculate Buy Score in candidate table"), value=True)

# -----------------------------
# Daily workflow
# -----------------------------
with tab0:
    st.subheader(tr("每日交易流程", "Daily Trading Workflow"))
    st.write(tr(
        "每天固定按这个顺序看：**市场 → 过热程度 → 板块 → 个股 → 催化剂 → 入场**。不要一打开就先找股票。",
        "Use the same order every day: **market → froth → sector → stock → catalyst → entry**. Do not start by hunting individual stocks."
    ))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(tr("### 1️⃣ 市场能不能做？", "### 1️⃣ Is the market tradable?"))
        st.write(tr("先看 SPY / QQQ + VIX。先判断 Risk-On、Mixed 还是 Risk-Off。", "Start with SPY / QQQ + VIX. Classify Risk-On, Mixed, or Risk-Off first."))
        st.markdown(tr("### 2️⃣ 今天应该多激进？", "### 2️⃣ How aggressive should I be?"))
        st.write(tr("看 Froth Score。越热，越应该缩仓、少追高、少用杠杆。", "Check the Froth Score. The hotter it is, the smaller and less aggressive you should be."))
        st.markdown(tr("### 3️⃣ 钱正在流向哪里？", "### 3️⃣ Where is capital rotating?"))
        st.write(tr("看板块相对 SPY 的 1D / 5D / 20D 强弱，优先找 Leading 或 Emerging。", "Compare 1D / 5D / 20D relative strength vs SPY; prefer Leading or Emerging groups."))
    with c2:
        st.markdown(tr("### 4️⃣ 哪些股票值得看？", "### 4️⃣ Which stocks deserve attention?"))
        st.write(tr("再跑 Momentum Scanner，筛涨幅、RVOL、突破和持续性。", "Then run the Momentum Scanner for gain, RVOL, breakout and persistence."))
        st.markdown(tr("### 5️⃣ 为什么涨？", "### 5️⃣ Why is it moving?"))
        st.write(tr("最后检查 catalyst。A/B 级催化剂优先；没新闻纯情绪拉升要更谨慎。", "Check the catalyst last. Prefer A/B-grade catalysts; be cautious with newsless spikes."))
        st.markdown(tr("### 6️⃣ 要不要进？", "### 6️⃣ Do I actually enter?"))
        st.write(tr("最后看 Buy Score 和 Entry Quality。杠杆 ETF 必须先看正股/底层资产评分，再决定要不要冲。", "Finish with Buy Score and Entry Quality. Leveraged ETFs must be judged from the underlying first."))

    st.info(tr("最重要的规则：**不要把顺序倒过来。先看市场，再看板块，最后才看股票。**", "Key rule: **do not reverse the order. Market first, sector second, stock last.**"))

    if st.button(tr("运行今日总览", "Run Today's Overview"), type="primary", use_container_width=True):
        with st.spinner(tr("正在计算今日环境...", "Calculating today's environment...")):
            regime_now = market_regime()
            spec_tickers = normalize_tickers(" ".join(SPEC_BASKET))
            leader_tickers = normalize_tickers(" ".join(LEADERS))
            all_froth = scan_universe(sorted(set(spec_tickers + leader_tickers)))
            spec = all_froth[all_froth["Ticker"].isin(spec_tickers)].copy()
            leaders_df = all_froth[all_froth["Ticker"].isin(leader_tickers)].copy()
            froth = froth_score(spec, leaders_df)
            rot, _ = rotation_dashboard()

        integrated_froth = None
        integrated_label = "N/A"
        if froth:
            regime_adj = regime_now["risk_adjustment"] if regime_now else 0
            complacency_adj = 0
            if regime_now and regime_now["complacency"] and (froth["Spec >10% today"] >= 10 or froth["Spec RVOL >=2"] >= 30):
                complacency_adj = 8
            integrated_froth = round(clamp(froth["score"] + regime_adj + complacency_adj), 1)
            integrated_label = "Normal" if integrated_froth < 30 else "Warm" if integrated_froth < 50 else "Hot" if integrated_froth < 70 else "Frothy" if integrated_froth < 85 else "Extreme"

        st.markdown(tr("## 今日结论", "## Today's Summary"))
        a,b,c,d = st.columns(4)
        if regime_now:
            a.metric(tr("市场环境", "Market Regime"), regime_now["regime"])
            b.metric("VIX", f"{regime_now['vix_level']:.2f}", delta=f"{regime_now['vix_day']:+.2f}%")
            c.metric(tr("SPY/QQQ 平均", "SPY/QQQ Average"), f"{regime_now['equity_day']:+.2f}%")
        if integrated_froth is not None:
            d.metric(tr("综合过热分数", "Integrated Froth"), f"{integrated_froth:.0f}/100", delta=integrated_label)

        if not rot.empty:
            lead = rot.iloc[0]; lag = rot.iloc[-1]
            r1,r2 = st.columns(2)
            r1.metric(tr("当前最强梯队", "Leading Group"), disp_group(lead["Group"]), delta=f"{lead['RS vs SPY 5D']:+.2f}% vs SPY / 5D")
            r2.metric(tr("当前最弱梯队", "Weakest Group"), disp_group(lag["Group"]), delta=f"{lag['RS vs SPY 5D']:+.2f}% vs SPY / 5D")

        st.markdown(tr("### 今天的执行建议", "### Execution Guidance"))
        if regime_now:
            if "Risk-Off" in regime_now["regime"]:
                advice = tr("降低追高、杠杆和高 Beta 暴露，先保护本金。", "Reduce chasing, leverage and high-beta exposure; preserve capital first.")
                st.error(advice)
            elif integrated_froth is not None and integrated_froth >= 70:
                advice = tr("市场投机温度较高：缩小仓位，减少追涨距离。", "Speculation is elevated: use smaller size and shorter chase distance.")
                st.warning(advice)
            elif regime_now["regime"] in ["Healthy Risk-On", "Risk-On"]:
                advice = tr("可以正常寻找多头 momentum，但仍先看强板块，再看个股。", "Normal long-momentum hunting is reasonable, but still check sector leadership before stocks.")
                st.success(advice)
            else:
                st.warning(tr("可以做，但要更挑剔；优先强板块中的 A/B 级 catalyst。", "Tradable, but be selective; prioritize A/B catalysts inside strong groups."))

        st.markdown(tr("### 每日 Checklist", "### Daily Checklist"))
        checks = [
            tr("市场环境不是明显 Risk-Off", "Market regime is not clearly Risk-Off"),
            tr("Froth 没有高到需要大幅降风险", "Froth is not high enough to require major de-risking"),
            tr("目标板块至少不是 Losing Leadership", "Target group is at least not Losing Leadership"),
            tr("个股 Momentum Score / RVOL 足够强，且 Buy Score 支持入场", "Momentum/RVOL are strong and Buy Score supports the setup"),
            tr("有能解释上涨的真实 catalyst", "A real catalyst explains the move"),
            tr("入场前已经想好止损和最大亏损", "Stop and maximum loss are defined before entry"),
        ]
        for x in checks:
            st.write(f"- {x}")

# -----------------------------
# Momentum scanner
# -----------------------------
with tab1:
    if st.button(tr("运行动量扫描", "Run Momentum Scanner"), type="primary", use_container_width=True):
        try:
            if universe_mode == "Core watchlist": tickers = DEFAULT_TICKERS
            elif universe_mode == "Custom tickers": tickers = normalize_tickers(custom)
            elif universe_mode == "S&P 500": tickers = load_sp500()
            elif universe_mode == "Nasdaq-100": tickers = load_nasdaq100()
            else: tickers = sorted(set(load_sp500() + load_nasdaq100()))
        except Exception as e:
            st.error(f"Universe error: {e}"); st.stop()

        progress = st.progress(0)
        df = scan_universe(tickers, progress)
        if df.empty:
            st.error(tr("没有返回行情数据。", "No market data returned."))
        else:
            filt = (df["Price"] >= min_price) & (df["Day %"] >= min_gain) & (df["Volume"] >= min_volume) & (df["RVOL"].fillna(0) >= min_rvol)
            if require_breakout: filt &= df["20D Breakout"]
            out = df.loc[filt].sort_values(["Momentum Score","RVOL","Day %"], ascending=False).head(top_n).copy()
            if out.empty:
                st.warning(tr("没有股票通过当前筛选。", "Nothing passed the current filters."))
                st.dataframe(df.sort_values("Momentum Score", ascending=False).head(20), use_container_width=True, hide_index=True)
            else:
                if include_buy_score:
                    with st.spinner(tr("正在计算 Buy Score（包括新闻、市场环境和板块）...", "Calculating Buy Scores (news, regime and sector)...")):
                        score_regime = market_regime()
                        score_rotation, _ = rotation_dashboard()
                        out = enrich_buy_scores(out, score_regime, score_rotation)
                st.subheader(tr("动量候选", "Momentum Candidates"))
                st.caption(tr("Momentum Score 看‘有多强’，Buy Score 看‘现在值不值得买’。杠杆 ETF 的 Buy Score 以正股/底层 proxy 为依据。", "Momentum Score measures strength; Buy Score measures setup quality now. Leveraged ETFs are scored from their underlying/proxy."))
                st.dataframe(localized_momentum(out), use_container_width=True, hide_index=True)
                st.download_button(tr("下载候选 CSV", "Download Candidates CSV"), out.to_csv(index=False).encode("utf-8"), "momentum_candidates.csv", "text/csv")
                st.subheader(tr("催化剂检查", "Catalyst Check"))
                st.caption(tr("自动分级只是关键词启发式，交易前必须读实际新闻。", "Automatic grading is only a keyword heuristic; read the actual news before trading."))
                for _, r in out.head(min(10,len(out))).iterrows():
                    info = product_info(r["Ticker"])
                    basis = info["underlying"]
                    news = get_news(basis); grade, reason = grade_news(news)
                    buy_txt = f" — Buy {float(r['Buy Score']):.1f}/100" if "Buy Score" in out.columns and not pd.isna(r.get("Buy Score", np.nan)) else ""
                    lev_txt = f" — {r.get('Leverage Verdict','')}" if r.get("Leverage Verdict","") else ""
                    with st.expander(f"{r['Ticker']} — Momentum {r['Momentum Score']:.1f}/100{buy_txt} — Catalyst {grade}{lev_txt}"):
                        if info["is_leveraged"]:
                            st.write(tr(f"评分依据：{basis}（正股/底层 proxy）", f"Score basis: {basis} (underlying/proxy)"))
                        st.write(reason)
                        if not news: st.write(tr("没有返回可用新闻。", "No usable headline returned."))
                        for item in news[:5]:
                            title=item["title"] or "(untitled)"
                            st.markdown(f"- [{title}]({item['url']}) — {item['publisher']}" if item["url"] else f"- {title} — {item['publisher']}")
    else:
        st.info(tr("在侧边栏设置条件，然后运行扫描。", "Set filters in the sidebar, then run the scanner."))

# -----------------------------
# Buy Score evaluator
# -----------------------------
with tabbuy:
    st.subheader(tr("Buy Score / 杠杆 ETF 正股评分", "Buy Score / Leveraged ETF Underlying Score"))
    st.write(tr(
        "输入任何股票或杠杆 ETF。普通股票直接评分；已识别的杠杆 ETF 会自动改用正股/底层 proxy 计算 Buy Score，再给杠杆结论。",
        "Enter any stock or leveraged ETF. Regular stocks are scored directly; recognized leveraged ETFs are scored from the underlying/proxy first, then receive a leverage verdict."
    ))
    st.caption(tr(
        "Buy Score 是 setup 质量分，不是上涨概率。Momentum 很高但已经过度延伸时，Buy Score 可以明显更低。",
        "Buy Score is a setup-quality score, not a probability of profit. A very strong but overextended move can have a much lower Buy Score."
    ))

    b1, b2 = st.columns([1,1])
    with b1:
        bs_ticker = st.text_input(tr("Ticker", "Ticker"), value="MRNX").upper().strip()
    with b2:
        bs_override = st.text_input(tr("底层资产手动覆盖（可选）", "Underlying override (optional)"), value="").upper().strip()

    if st.button(tr("计算 Buy Score", "Calculate Buy Score"), type="primary", use_container_width=True):
        if not bs_ticker:
            st.warning(tr("请输入 ticker。", "Enter a ticker."))
        else:
            info = product_info(bs_ticker, bs_override)
            needed = sorted(set([bs_ticker, info["underlying"]]))
            with st.spinner(tr("正在读取行情、新闻和市场环境...", "Loading price, news and market regime...")):
                px = scan_universe(needed)

            # Price data is required before doing the more expensive market/rotation calls.
            # If Yahoo is temporarily unavailable, fail gracefully instead of throwing KeyError.
            if px.empty:
                st.error(tr(
                    f"没有拿到 {info['underlying']} 的行情数据。可能是 Yahoo/yfinance 暂时没有返回数据；请过一会儿重试。若这是杠杆 ETF，也可以手动填写正确的底层 ticker。",
                    f"No market data was returned for {info['underlying']}. Yahoo/yfinance may be temporarily unavailable; try again shortly. For a leveraged ETF, you can also enter the correct underlying ticker manually."
                ))
            else:
                regime_bs = market_regime()
                rotation_bs, _ = rotation_dashboard()
                product_rows = px[px["Ticker"] == bs_ticker]
                basis_rows = px[px["Ticker"] == info["underlying"]]
                if basis_rows.empty:
                    st.error(tr(
                        f"拿到了部分行情，但没有拿到 {info['underlying']}。请检查 ticker，或稍后重试。",
                        f"Partial market data was returned, but {info['underlying']} was missing. Check the ticker or try again shortly."
                    ))
                else:
                    basis = basis_rows.iloc[0]
                    news = get_news(info["underlying"])
                    grade, reason = grade_news(news)
                    score = buy_score(basis, grade, regime_bs, rotation_bs, info["underlying"], info["direction"])
                    product_day = float(product_rows.iloc[0]["Day %"]) if not product_rows.empty else np.nan
                    verdict = leverage_verdict(score, info["leverage"], product_day) if info["is_leveraged"] else "—"

                    a,b,c,d = st.columns(4)
                    a.metric(tr("Buy Score" if info["direction"] == 1 else "Directional Score", "Buy Score" if info["direction"] == 1 else "Directional Score"), f"{score['score']:.0f}/100", delta=score["label"])
                    b.metric(tr("Entry Quality", "Entry Quality"), f"{score['entry_quality']:.0f}/100")
                    c.metric(tr("Catalyst", "Catalyst"), grade)
                    d.metric(tr("Momentum", "Momentum"), f"{float(basis['Momentum Score']):.0f}/100")

                    if info["is_leveraged"]:
                        proxy_word = tr("底层 proxy", "underlying proxy") if info["proxy"] else tr("正股", "underlying stock")
                        st.info(tr(
                            f"{bs_ticker} 被识别为约 {info['leverage']}× {'反向' if info['direction']==-1 else '做多'}产品。评分依据：{info['underlying']}（{proxy_word}）。杠杆结论：{verdict}",
                            f"{bs_ticker} is recognized as an approximately {info['leverage']}x {'inverse' if info['direction']==-1 else 'long'} product. Score basis: {info['underlying']} ({proxy_word}). Leverage verdict: {verdict}"
                        ))
                        if verdict == "DO NOT CHASE":
                            st.warning(tr("底层可能仍然很强，但当前入场质量或杠杆产品本身的单日延伸度太差，不建议追。", "The underlying may still be strong, but entry quality or the leveraged product's one-day extension is too poor to chase."))
                    else:
                        st.info(tr(f"{bs_ticker} 直接按自身行情评分。当前 Setup：{score['label']}", f"{bs_ticker} is scored directly. Current setup: {score['label']}"))

                    comp = pd.DataFrame([
                        [tr("趋势 / 相对强弱", "Trend / Relative Strength"), score["components"]["Trend / Relative Strength"], 20],
                        [tr("催化剂", "Catalyst"), score["components"]["Catalyst"], 20],
                        [tr("入场质量", "Entry Quality"), score["components"]["Entry Quality"], 20],
                        [tr("RVOL / 成交量", "RVOL / Volume"), score["components"]["RVOL / Volume"], 15],
                        [tr("大盘环境", "Market Regime"), score["components"]["Market Regime"], 10],
                        [tr("板块环境", "Sector Regime"), score["components"]["Sector Regime"], 5],
                        [tr("风险收益", "Risk / Reward"), score["components"]["Risk / Reward"], 10],
                    ], columns=[tr("项目", "Component"), tr("得分", "Score"), tr("满分", "Max")])
                    st.markdown(tr("#### 分数组成", "#### Score Breakdown"))
                    st.dataframe(comp, use_container_width=True, hide_index=True)

                    q1,q2,q3,q4 = st.columns(4)
                    q1.metric(tr("底层 1D", "Underlying 1D"), f"{float(basis['Day %']):+.2f}%")
                    q2.metric(tr("底层 5D", "Underlying 5D"), f"{float(basis['5D %']):+.2f}%")
                    q3.metric("RVOL", f"{float(basis['RVOL']):.2f}x" if not np.isnan(basis['RVOL']) else "N/A")
                    q4.metric("ATR %", f"{float(basis['ATR %']):.2f}%" if not np.isnan(basis['ATR %']) else "N/A")

                    st.markdown(tr("#### 催化剂新闻", "#### Catalyst News"))
                    st.write(reason)
                    if not news:
                        st.write(tr("没有返回可用新闻。", "No usable headline returned."))
                    for item in news[:5]:
                        title = item["title"] or "(untitled)"
                        st.markdown(f"- [{title}]({item['url']}) — {item['publisher']}" if item["url"] else f"- {title} — {item['publisher']}")

                    st.caption(tr(
                        "阈值：85+ Strong Buy Setup；75–84 Buy/Tactical；65–74 Watch；50–64 Weak；<50 Avoid。3× 产品要求更高，而且每日重置会让多日收益偏离简单倍数。",
                        "Thresholds: 85+ Strong Buy Setup; 75–84 Buy/Tactical; 65–74 Watch; 50–64 Weak; <50 Avoid. 3x products face a higher hurdle and daily reset can make multi-day returns diverge from a simple multiple."
                    ))

# -----------------------------
# Froth gauge
# -----------------------------
with tab2:
    st.subheader(tr("市场过热指标", "Market Froth Gauge"))
    st.write(tr("判断高 Beta / 投机股是否正在广泛跑赢成熟龙头，并伴随异常成交量和突破。", "Checks whether high-beta/speculative names are broadly outperforming established leaders with unusual volume and breakouts."))
    st.caption(tr("它是风险温度计，不是崩盘计时器。", "It is a risk-temperature gauge, not a crash timer."))
    custom_spec = st.text_area(tr("投机代理篮子", "Speculation Proxy Basket"), value=" ".join(SPEC_BASKET), height=100)
    custom_leaders = st.text_area(tr("龙头参考篮子", "Leadership Reference Basket"), value=" ".join(LEADERS), height=70)
    if st.button(tr("计算过热分数", "Calculate Froth Score"), type="primary", use_container_width=True):
        spec_tickers=normalize_tickers(custom_spec); leader_tickers=normalize_tickers(custom_leaders)
        all_df=scan_universe(sorted(set(spec_tickers+leader_tickers)), st.progress(0))
        spec=all_df[all_df["Ticker"].isin(spec_tickers)].copy(); leaders_df=all_df[all_df["Ticker"].isin(leader_tickers)].copy()
        result=froth_score(spec, leaders_df)
        if result is None:
            st.error(tr("数据不足。", "Not enough market data."))
        else:
            regime_now=market_regime(); base=result["score"]; adj=regime_now["risk_adjustment"] if regime_now else 0
            complacency=8 if regime_now and regime_now["complacency"] and (result["Spec >10% today"]>=10 or result["Spec RVOL >=2"]>=30) else 0
            integrated=round(clamp(base+adj+complacency),1)
            label="Normal" if integrated<30 else "Warm" if integrated<50 else "Hot" if integrated<70 else "Frothy" if integrated<85 else "Extreme"
            x,y,z=st.columns(3)
            x.metric(tr("综合过热分数", "Integrated Froth Score"), f"{integrated:.0f}/100", delta=f"{integrated-base:+.0f} regime adj.")
            y.metric(tr("状态", "Regime"), label)
            z.metric(tr("投机股 vs 龙头（今日）", "Spec vs Leaders Today"), f"{result['Day spread']:+.2f}%")
            st.progress(integrated/100)
            if regime_now:
                st.markdown(tr("#### VIX / 指数叠加", "#### VIX / Index Overlay"))
                a,b,c=st.columns(3)
                a.metric(tr("市场环境", "Market Regime"), regime_now["regime"]); b.metric("VIX", f"{regime_now['vix_level']:.2f}", delta=f"{regime_now['vix_day']:+.2f}%"); c.metric("SPY/QQQ", f"{regime_now['equity_day']:+.2f}%")
                st.write(regime_now["meaning"])
            hottest=spec.sort_values(["Momentum Score","Day %"],ascending=False).head(15)
            st.markdown(tr("#### 最热投机股", "#### Hottest Speculative Names")); st.dataframe(hottest[["Ticker","Price","Day %","5D %","RVOL","20D Breakout","Momentum Score"]], use_container_width=True, hide_index=True)

# -----------------------------
# VIX regime
# -----------------------------
with tab3:
    st.subheader(tr("VIX + 市场环境", "VIX + Market Regime"))
    st.write(tr("把 SPY/QQQ 方向与 VIX 方向组合起来，不把“VIX 下跌”简单等于“利好”。", "Combine SPY/QQQ direction with VIX direction instead of treating falling VIX as automatically bullish."))
    if st.button(tr("计算市场环境", "Calculate Market Regime"), type="primary", use_container_width=True):
        r=market_regime()
        if not r: st.error(tr("数据不足。", "Not enough data."))
        else:
            a,b,c,d=st.columns(4)
            a.metric(tr("状态", "Regime"), r["regime"]); b.metric("VIX",f"{r['vix_level']:.2f}",delta=f"{r['vix_day']:+.2f}%"); c.metric("SPY",f"{r['spy_day']:+.2f}%"); d.metric("QQQ",f"{r['qqq_day']:+.2f}%")
            st.info(r["meaning"])
            combo=pd.DataFrame([
                ["SPY/QQQ ↑","VIX ↓","Healthy Risk-On",tr("最适合普通多头 momentum，但仍要有有效 setup。","Best for ordinary long momentum, but still require a valid setup.")],
                ["SPY/QQQ ↑","VIX ↑","Conflicted Rally",tr("上涨同时保护需求上升，要更挑剔。","Rally plus hedging demand; be more selective.")],
                ["SPY/QQQ ↓","VIX ↑","Risk-Off",tr("不适合激进高 Beta 多头。","Poor backdrop for aggressive high-beta longs.")],
                ["SPY/QQQ ≈ flat","VIX ↓","Compression",tr("环境变平静，但要等价格确认。","Calmer conditions, but wait for price confirmation.")],
                ["SPY/QQQ ↓","VIX ↓","Orderly Weakness",tr("弱但不是恐慌。","Weak, but not a volatility shock.")],
            ],columns=[tr("指数","Index"),"VIX",tr("环境","Regime"),tr("解释","Interpretation")])
            st.dataframe(combo,use_container_width=True,hide_index=True)

# -----------------------------
# Sector rotation
# -----------------------------
with tab4:
    st.subheader(tr("板块 / 行业领导力轮动", "Sector / Industry Leadership Rotation"))
    st.write(tr("同时比较行业、板块、大小盘和风格组与 SPY 的相对强弱；主要主题尽量使用多个 ETF proxy。", "Compare sector, industry, size and style groups vs SPY; major themes use multiple liquid ETF proxies where practical."))
    st.caption(tr("一起看 1 日、5 日、20 日，区分单日回调和真正失去领导力。", "Use 1D, 5D and 20D together to separate a one-day pullback from a real loss of leadership."))
    if st.button(tr("计算轮动面板", "Calculate Rotation Dashboard"), type="primary", use_container_width=True):
        rot,_=rotation_dashboard()
        if rot.empty: st.error(tr("数据不足。", "Not enough data."))
        else:
            lead=rot.iloc[0]; lag=rot.iloc[-1]
            a,b,c=st.columns(3)
            a.metric(tr("当前最强", "Current Leader"), disp_group(lead["Group"]), delta=f"{lead['RS vs SPY 5D']:+.2f}% vs SPY / 5D")
            b.metric(tr("当前最弱", "Weakest Group"), disp_group(lag["Group"]), delta=f"{lag['RS vs SPY 5D']:+.2f}% vs SPY / 5D")
            c.metric(tr("跟踪梯队", "Groups Tracked"), len(rot))
            st.markdown(tr("#### 领导力地图", "#### Leadership Map")); st.dataframe(localized_rotation(rot),use_container_width=True,hide_index=True)
            st.markdown(tr("#### 如何理解“半导体跌但大盘还好”", "#### How to read a semiconductor-style divergence"))
            st.write(tr(
                "如果 VIX 模块显示 Healthy Risk-On，但半导体显示 Deteriorating / Losing Leadership，更像板块轮动。如果半导体、软件、Growth、小盘同时恶化，而防御板块领涨，才更像市场环境整体转弱。",
                "If the VIX tab says Healthy Risk-On but Semiconductors show Deteriorating / Losing Leadership, that is more consistent with sector rotation. If Semiconductors, Software, Growth and Small Caps all deteriorate while defensives lead, evidence of a broad regime change is much stronger."
            ))
            st.download_button(tr("下载轮动 CSV", "Download Rotation CSV"), rot.to_csv(index=False).encode("utf-8"), "sector_rotation_snapshot.csv", "text/csv")

st.divider()
st.caption(tr(
    "限制：Yahoo/yfinance 数据可能延迟或不完整。VIX 是隐含波动率，不是方向预测。所有分数都是启发式，不是投资建议或崩盘预测。",
    "Limitations: Yahoo/yfinance data may be delayed or incomplete. VIX is implied volatility, not a directional forecast. Scores are heuristics, not investment advice or crash predictors."
))
