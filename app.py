
import re
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Momentum Scanner V6.7 Freshness Guard", page_icon="📈", layout="wide")


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


@st.cache_data(ttl=45, show_spinner=False)
def download_intraday(ticker):
    """
    Near-real-time intraday overlay.
    Yahoo/yfinance is not exchange-direct market data and can still be delayed.
    1-minute bars are used when available, including pre/post market.
    """
    t = str(ticker).upper().strip()
    try:
        df = yf.download(
            tickers=t,
            period="1d",
            interval="1m",
            auto_adjust=False,
            prepost=True,
            progress=False,
            threads=False,
        )
        if df is None or df.empty:
            return pd.DataFrame()
        # Normalize current yfinance MultiIndex behavior for one ticker.
        return ticker_frame(df, t, 1) if "ticker_frame" in globals() else df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=45, show_spinner=False)
def download_intraday_5d(ticker):
    """
    5-minute history used for a rough same-time intraday volume comparison.
    This is intentionally lightweight and is only called for focused Buy Score checks.
    """
    t = str(ticker).upper().strip()
    try:
        df = yf.download(
            tickers=t,
            period="5d",
            interval="5m",
            auto_adjust=False,
            prepost=False,
            progress=False,
            threads=False,
        )
        if df is None or df.empty:
            return pd.DataFrame()
        return ticker_frame(df, t, 1) if "ticker_frame" in globals() else df
    except Exception:
        return pd.DataFrame()


def _session_dates(index):
    try:
        idx = pd.DatetimeIndex(index)
        if idx.tz is not None:
            idx = idx.tz_convert("America/New_York")
        return pd.Series(idx.date, index=idx)
    except Exception:
        return pd.Series(dtype=object)


def live_overlay_metrics(daily_metrics, ticker):
    """
    Overlay intraday fields onto daily metrics.

    Daily bars still determine structural fields such as 20D high/low and ATR.
    Intraday data updates:
      - Price
      - Day %
      - Day High / Day Low
      - Close Location
      - distance vs 20D high/low/support
      - 20D breakout/breakdown status
      - approximate same-time intraday RVOL when 5m history is available
      - live timestamp
    """
    if daily_metrics is None:
        return None, {"live": False, "reason": "No daily metrics"}

    m = dict(daily_metrics)
    intraday = download_intraday(ticker)
    if intraday is None or intraday.empty or "Close" not in intraday.columns:
        m["Live Data"] = False
        m["Live Timestamp"] = ""
        return m, {"live": False, "reason": "No intraday data"}

    intraday = intraday.dropna(subset=["Close"]).copy()
    if intraday.empty:
        m["Live Data"] = False
        m["Live Timestamp"] = ""
        return m, {"live": False, "reason": "No valid intraday close"}

    price = float(intraday["Close"].iloc[-1])
    day_high = float(intraday["High"].max()) if "High" in intraday else price
    day_low = float(intraday["Low"].min()) if "Low" in intraday else price

    # Previous close from the daily history-derived fields.
    old_price = float(daily_metrics.get("Price", price))
    old_day_pct = float(daily_metrics.get("Day %", 0) or 0)
    prev_close = old_price / (1 + old_day_pct / 100) if (1 + old_day_pct / 100) != 0 else old_price

    day_pct = (price / prev_close - 1) * 100 if prev_close else 0.0
    clv = (price - day_low) / (day_high - day_low) if day_high > day_low else 0.5

    high20 = float(m.get("20D High Price", price))
    low20 = float(m.get("20D Low Price", price))
    support10 = float(m.get("10D Support Price", low20))

    m["Price"] = price
    m["Day %"] = day_pct
    m["Day High"] = day_high
    m["Day Low"] = day_low
    m["Close Location"] = clv
    m["vs 20D High %"] = (price / high20 - 1) * 100 if high20 else np.nan
    m["vs 20D Low %"] = (price / low20 - 1) * 100 if low20 else np.nan
    m["vs 10D Support %"] = (price / support10 - 1) * 100 if support10 else np.nan
    m["20D Breakout"] = bool(price > high20)
    m["20D Breakdown"] = bool(price < low20)

    # Recompute current ATR% against today's live price while keeping ATR dollars
    # approximately anchored to the daily ATR estimate.
    old_atr_pct = float(daily_metrics.get("ATR %", np.nan))
    if not np.isnan(old_atr_pct) and old_price:
        atr_dollars = old_price * old_atr_pct / 100
        m["ATR %"] = atr_dollars / price * 100 if price else old_atr_pct

    # Approximate same-time RVOL using 5-minute cumulative regular-session volume.
    # If unavailable, retain the daily RVOL estimate.
    try:
        hist5 = download_intraday_5d(ticker)
        if hist5 is not None and not hist5.empty and "Volume" in hist5.columns:
            idx = pd.DatetimeIndex(hist5.index)
            if idx.tz is not None:
                idx_et = idx.tz_convert("America/New_York")
            else:
                idx_et = idx
            tmp = hist5.copy()
            tmp["_date"] = idx_et.date
            tmp["_time"] = idx_et.time
            unique_dates = list(pd.unique(tmp["_date"]))
            if len(unique_dates) >= 2:
                today = unique_dates[-1]
                prev_dates = unique_dates[:-1]
                today_rows = tmp[tmp["_date"] == today]
                if not today_rows.empty:
                    cutoff = today_rows["_time"].iloc[-1]
                    current_cum = float(today_rows["Volume"].fillna(0).sum())
                    prev_cums = []
                    for d in prev_dates:
                        drows = tmp[(tmp["_date"] == d) & (tmp["_time"] <= cutoff)]
                        if not drows.empty:
                            prev_cums.append(float(drows["Volume"].fillna(0).sum()))
                    if prev_cums and np.mean(prev_cums) > 0:
                        m["RVOL"] = current_cum / float(np.mean(prev_cums))
                        m["Volume"] = current_cum
    except Exception:
        pass

    try:
        ts = pd.Timestamp(intraday.index[-1])
        if ts.tzinfo is not None:
            ts = ts.tz_convert("America/New_York")
        timestamp = ts.strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        timestamp = str(intraday.index[-1])

    m["Live Data"] = True
    m["Live Timestamp"] = timestamp
    return m, {"live": True, "timestamp": timestamp, "timestamp_obj": intraday.index[-1]}


def recompute_momentum_score(m):
    """Recalculate momentum after the live overlay."""
    pct = float(m.get("Day %", 0) or 0)
    rvol = m.get("RVOL", np.nan)
    clv = m.get("Close Location", np.nan)
    score = 0
    score += min(max(pct, 0), 25) * 1.4
    if not np.isnan(rvol):
        score += min(max(float(rvol) - 1, 0), 5) * 7
    score += 15 if bool(m.get("20D Breakout", False)) else 0
    score += 5 if bool(m.get("50D Breakout", False)) else 0
    if not np.isnan(clv):
        score += max(min(float(clv), 1), 0) * 10
    m["Momentum Score"] = min(round(score, 1), 100)
    return m


def freshness_status(live_meta, max_age_minutes=20):
    """
    Strict stale-data guard.

    During US extended trading hours (04:00-20:00 ET, weekdays), live data must:
      - be from today's ET date
      - be no older than max_age_minutes

    Outside those hours, the market is treated as closed and daily/last-session
    data may be used only when explicitly labeled as such.
    """
    now_et = pd.Timestamp.now(tz="America/New_York")
    weekday = now_et.weekday() < 5
    minutes_now = now_et.hour * 60 + now_et.minute
    extended_open = weekday and (4 * 60 <= minutes_now < 20 * 60)

    if not live_meta or not live_meta.get("live"):
        return {
            "ok": not extended_open,
            "market_open": extended_open,
            "stale": extended_open,
            "age_minutes": None,
            "message": "No live intraday timestamp available."
        }

    ts_raw = live_meta.get("timestamp_obj") or live_meta.get("timestamp")
    try:
        ts = pd.Timestamp(ts_raw)
        if ts.tzinfo is None:
            ts = ts.tz_localize("America/New_York")
        else:
            ts = ts.tz_convert("America/New_York")
    except Exception:
        return {
            "ok": not extended_open,
            "market_open": extended_open,
            "stale": extended_open,
            "age_minutes": None,
            "message": "Could not parse live timestamp."
        }

    age_minutes = max((now_et - ts).total_seconds() / 60.0, 0.0)
    same_date = ts.date() == now_et.date()

    if extended_open:
        ok = same_date and age_minutes <= max_age_minutes
        return {
            "ok": ok,
            "market_open": True,
            "stale": not ok,
            "age_minutes": age_minutes,
            "timestamp": ts,
            "message": (
                f"Live data age {age_minutes:.1f} min."
                if ok else
                f"Stale intraday data: last bar {ts.strftime('%Y-%m-%d %H:%M:%S %Z')} "
                f"({age_minutes:.1f} min old)."
            )
        }

    return {
        "ok": True,
        "market_open": False,
        "stale": False,
        "age_minutes": age_minutes,
        "timestamp": ts,
        "message": f"Market closed; latest intraday bar {ts.strftime('%Y-%m-%d %H:%M:%S %Z')}."
    }


def ticker_frame(blob, ticker, n_tickers):
    """
    Normalize yfinance output for both single- and multi-ticker downloads.

    Recent yfinance versions may return MultiIndex columns even when only one
    ticker is requested. Older code assumed a single ticker always had flat
    OHLCV columns, which caused valid symbols such as BE to appear as if no
    market data existed.
    """
    if blob is None or getattr(blob, "empty", True):
        return pd.DataFrame()

    df = blob.copy()

    # yfinance can return:
    #   MultiIndex level 0 = ticker, level 1 = Price
    # or
    #   MultiIndex level 0 = Price, level 1 = ticker
    if isinstance(df.columns, pd.MultiIndex):
        extracted = None

        # Try each column level for the requested ticker.
        for level in range(df.columns.nlevels):
            values = set(str(x) for x in df.columns.get_level_values(level))
            if ticker in values:
                try:
                    extracted = df.xs(ticker, axis=1, level=level, drop_level=True).copy()
                    break
                except Exception:
                    pass

        # If only one ticker was requested and the ticker label is not exposed
        # in a predictable level, identify the OHLCV level and flatten to it.
        if extracted is None and n_tickers == 1:
            price_fields = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
            for level in range(df.columns.nlevels):
                vals = set(str(x) for x in df.columns.get_level_values(level))
                if len(price_fields.intersection(vals)) >= 3:
                    try:
                        # If another level contains only one repeated label,
                        # dropping it leaves ordinary OHLCV columns.
                        other_levels = [i for i in range(df.columns.nlevels) if i != level]
                        candidate = df.copy()
                        # Rebuild columns from the detected price-field level.
                        candidate.columns = [str(x) for x in df.columns.get_level_values(level)]
                        extracted = candidate
                        break
                    except Exception:
                        pass

        if extracted is None:
            return pd.DataFrame()
        df = extracted

    # Final defensive cleanup.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join(str(x) for x in col if str(x) not in ("", "None"))
            for col in df.columns
        ]

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

    day_high = float(df["High"].dropna().iloc[-1]) if "High" in df and len(df["High"].dropna()) else last
    day_low = float(df["Low"].dropna().iloc[-1]) if "Low" in df and len(df["Low"].dropna()) else last

    return {
        "Price": last, "Day %": pct, "5D %": ret5, "20D %": ret20,
        "RVOL": rvol, "Volume": v_last,
        "20D Breakout": breakout20, "50D Breakout": breakout50, "20D Breakdown": breakdown20,
        "20D High Price": prior20, "20D Low Price": prior20low, "10D Support Price": prior10low,
        "Day High": day_high, "Day Low": day_low,
        "vs 20D High %": dist20, "vs 20D Low %": dist20low, "vs 10D Support %": dist10support,
        "ATR %": atr_pct, "Close Location": clv,
        "Momentum Score": score,
    }

SCAN_COLUMNS = [
    "Ticker", "Price", "Day %", "5D %", "20D %", "RVOL", "Volume",
    "20D Breakout", "50D Breakout", "20D Breakdown",
    "20D High Price", "20D Low Price", "10D Support Price", "Day High", "Day Low",
    "vs 20D High %", "vs 20D Low %", "vs 10D Support %", "ATR %", "Close Location",
    "Momentum Score", "Live Data", "Live Timestamp",
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


# ETF catalyst logic:
# Industry/sector ETFs should not be penalized just because they do not publish
# company-style earnings/FDA/contract headlines. Their catalyst quality is
# inferred from group leadership, breadth, relative strength and volume.
ETF_GROUP_HINTS = {
    **{x:"Technology" for x in "XLK VGT".split()},
    **{x:"Semiconductors" for x in "SMH SOXX XSD".split()},
    **{x:"Software / Cloud" for x in "IGV SKYY CLOU".split()},
    **{x:"Cybersecurity" for x in "CIBR HACK BUG".split()},
    **{x:"Communication / Internet" for x in "XLC FDN".split()},
    **{x:"Consumer Discretionary" for x in "XLY RTH".split()},
    **{x:"Consumer Staples" for x in "XLP".split()},
    **{x:"Financials / Banks" for x in "XLF KBE KRE".split()},
    **{x:"Industrials" for x in "XLI PAVE".split()},
    **{x:"Aerospace / Defense" for x in "ITA XAR".split()},
    **{x:"Energy" for x in "XLE XOP OIH".split()},
    **{x:"Utilities" for x in "XLU".split()},
    **{x:"Health Care" for x in "XLV IHI".split()},
    **{x:"Biotech" for x in "XBI IBB".split()},
    **{x:"Materials / Metals" for x in "XLB XME".split()},
    **{x:"Real Estate" for x in "XLRE IYR".split()},
    **{x:"Homebuilders" for x in "XHB ITB".split()},
    **{x:"Transportation" for x in "IYT".split()},
    **{x:"Small Caps" for x in "IWM IJR".split()},
    **{x:"Mid Caps" for x in "MDY".split()},
    **{x:"Growth" for x in "QQQ SPYG IWF".split()},
    **{x:"Value" for x in "SPYV IWD".split()},
}
BROAD_MARKET_ETFS = {"SPY", "DIA"}

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
    group = TICKER_GROUP_HINTS.get(str(ticker).upper()) or ETF_GROUP_HINTS.get(str(ticker).upper())
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



def resolve_catalyst(ticker, metrics, rotation=None, regime=None):
    """
    Return a structured catalyst signal.

    Stocks: company-news heuristic.
    Sector/industry ETFs: group leadership + breadth + RVOL + persistence.
    Broad-market ETFs: market regime + ETF trend/breakout.
    """
    t = str(ticker).upper().strip()

    if t in ETF_GROUP_HINTS:
        group = ETF_GROUP_HINTS[t]
        points = 7.0
        details = []

        if rotation is not None and not rotation.empty:
            hit = rotation[rotation["Group"] == group]
            if not hit.empty:
                r = hit.iloc[0]
                leadership = float(r.get("Leadership Score", 50))
                breadth = float(r.get("20D Breakout Breadth %", 0))
                rvol = float(r.get("Median RVOL", 1))
                rs5 = float(r.get("RS vs SPY 5D", 0))
                state = str(r.get("State", "Mixed / Transition"))

                # 0..8 leadership contribution
                points += clamp((leadership - 35) / 65 * 8, 0, 8)
                # 0..3 breakout breadth
                points += clamp(breadth / 100 * 3, 0, 3)
                # 0..2 volume confirmation
                points += clamp((rvol - 0.8) / 1.7 * 2, 0, 2)

                # Persistence / state adjustment.
                if state == "Leading":
                    points += 2
                elif state == "Emerging / Improving":
                    points += 1.5
                elif state == "Pullback in Leader":
                    points += 0.5
                elif state == "Deteriorating":
                    points -= 2
                elif state == "Losing Leadership":
                    points -= 4

                if rs5 > 2:
                    points += 1
                elif rs5 < -2:
                    points -= 1

                details.append(
                    f"{group}: {state}; leadership {leadership:.0f}/100; "
                    f"20D breakout breadth {breadth:.0f}%; median RVOL {rvol:.2f}x; "
                    f"5D RS vs SPY {rs5:+.2f}%."
                )

        # ETF itself can add confirmation but not dominate the score.
        if bool(metrics.get("20D Breakout", False)):
            points += 1.5
        day = float(metrics.get("Day %", 0) or 0)
        rvol_self = metrics.get("RVOL", np.nan)
        if not np.isnan(rvol_self) and rvol_self >= 1.5:
            points += 1
        if day <= -4:
            points -= 1

        points = round(clamp(points, 2, 20), 1)
        label = "ETF-A" if points >= 16 else "ETF-B" if points >= 12 else "ETF-C" if points >= 8 else "ETF-Weak"
        reason = "ETF catalyst uses industry/sector leadership, breadth, relative strength and volume rather than company-news headlines."
        if details:
            reason += " " + " ".join(details)
        return {"label": label, "points": points, "reason": reason, "news": [], "kind": "ETF"}

    if t in BROAD_MARKET_ETFS:
        # Broad ETFs use market regime + own trend instead of company news.
        regime_name = regime.get("regime", "Neutral / Mixed") if regime else "Neutral / Mixed"
        regime_points = {
            "Healthy Risk-On": 15, "Risk-On": 14, "Compression / Watch Breakouts": 12,
            "Neutral / Mixed": 10, "Conflicted Rally": 9, "Orderly Weakness": 7,
            "Risk-Off Lean": 5, "Risk-Off": 3,
        }.get(regime_name, 9)
        points = float(regime_points)
        if bool(metrics.get("20D Breakout", False)):
            points += 2
        if float(metrics.get("5D %", 0) or 0) > 2:
            points += 1
        points = round(clamp(points, 2, 20), 1)
        label = "ETF-A" if points >= 16 else "ETF-B" if points >= 12 else "ETF-C" if points >= 8 else "ETF-Weak"
        return {
            "label": label, "points": points,
            "reason": f"Broad-market ETF catalyst is based on market regime ({regime_name}) plus ETF trend/breakout confirmation.",
            "news": [], "kind": "ETF"
        }

    news = get_news(t)
    grade, reason = grade_news(news)
    return {
        "label": grade,
        "points": CATALYST_POINTS.get(grade, 4.0),
        "reason": reason,
        "news": news,
        "kind": "Stock"
    }


def buy_score(m, catalyst_grade="?", regime=None, rotation=None, ticker="", direction=1):
    trend = _trend_component(m, direction)
    if isinstance(catalyst_grade, dict):
        catalyst = float(catalyst_grade.get("points", 4.0))
    else:
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



def entry_plan(m, score_result=None, direction=1):
    """
    Build a volatility-aware reference entry plan from the underlying asset.
    This is intentionally a zone/trigger framework rather than a single magic price.
    """
    try:
        price = float(m["Price"])
    except Exception:
        return None

    atr_pct = float(m.get("ATR %", np.nan))
    atr = price * atr_pct / 100 if not np.isnan(atr_pct) and atr_pct > 0 else price * 0.035

    day_high = float(m.get("Day High", price))
    day_low = float(m.get("Day Low", price))
    high20 = float(m.get("20D High Price", price))
    low20 = float(m.get("20D Low Price", price))
    support10 = float(m.get("10D Support Price", low20))
    entry_q = float(score_result.get("entry_quality", 50)) if score_result else 50
    day_pct = float(m.get("Day %", 0))

    # Wider-volatility names get a tighter "max chase" allowance in ATR terms.
    chase_mult = 0.22 if atr_pct >= 10 else 0.30 if atr_pct >= 6 else 0.38

    if direction == 1:
        # If already breaking out, favor a retest of the old 20D high.
        if bool(m.get("20D Breakout", False)):
            anchor = high20
            zone_low = max(anchor - 0.10 * atr, price - 0.80 * atr)
            zone_high = min(anchor + 0.30 * atr, price)
            setup = "Breakout retest"
        else:
            # Otherwise use the nearer of short-term support and a volatility pullback.
            vol_floor = price - 0.60 * atr
            anchor = max(support10, vol_floor)
            zone_low = min(anchor, price)
            # Strong entry quality can include current price; weak/extended setups demand a pullback.
            zone_high = price if entry_q >= 72 and day_pct < 6 else max(zone_low, price - 0.15 * atr)
            setup = "Pullback / support"

        trigger = max(day_high, high20) + 0.05 * atr
        max_chase = trigger + chase_mult * atr
        stop_ref = max(0.01, zone_low - 0.45 * atr)

        # Avoid inverted/degenerate ranges.
        zone_low, zone_high = sorted([zone_low, zone_high])

        return {
            "setup": setup,
            "zone_low": round(zone_low, 2),
            "zone_high": round(zone_high, 2),
            "breakout_trigger": round(trigger, 2),
            "max_chase": round(max_chase, 2),
            "stop_ref": round(stop_ref, 2),
            "basis_price": round(price, 2),
            "atr_dollars": round(atr, 2),
        }

    # Bearish / inverse-product logic: mirror around resistance and breakdown.
    resistance = min(high20, price + 0.60 * atr) if high20 >= price else price + 0.35 * atr
    zone_low = price if entry_q >= 72 and day_pct > -6 else min(price + 0.15 * atr, resistance)
    zone_high = max(resistance, zone_low)
    trigger = min(day_low, low20) - 0.05 * atr
    max_chase = trigger - chase_mult * atr
    stop_ref = zone_high + 0.45 * atr
    return {
        "setup": "Resistance / breakdown",
        "zone_low": round(zone_low, 2),
        "zone_high": round(zone_high, 2),
        "breakout_trigger": round(trigger, 2),
        "max_chase": round(max_chase, 2),
        "stop_ref": round(stop_ref, 2),
        "basis_price": round(price, 2),
        "atr_dollars": round(atr, 2),
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
    cols = [
        "Catalyst Grade", "Buy Score", "Entry Quality", "Setup", "Score Basis", "Leverage Verdict",
        "Entry Zone", "Breakout Trigger", "Max Chase", "Stop Ref"
    ]
    for c in cols:
        result[c] = "" if c in ["Catalyst Grade","Setup","Score Basis","Leverage Verdict","Entry Zone"] else np.nan

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
        if LIVE_MODE and len(result) <= 25:
            try:
                live_basis, _live_meta = live_overlay_metrics(dict(basis_metrics), basis_ticker)
                _fresh = freshness_status(_live_meta, MAX_LIVE_AGE)
                if live_basis is not None and _fresh.get("ok"):
                    basis_metrics = recompute_momentum_score(live_basis)
                elif STRICT_FRESHNESS and _fresh.get("market_open"):
                    result.at[idx, "Setup"] = "STALE DATA"
                    result.at[idx, "Score Basis"] = basis_ticker
                    continue
            except Exception:
                if STRICT_FRESHNESS:
                    result.at[idx, "Setup"] = "LIVE DATA ERROR"
                    result.at[idx, "Score Basis"] = basis_ticker
                    continue
        catalyst_signal = resolve_catalyst(basis_ticker, basis_metrics, rotation, regime)
        score = buy_score(basis_metrics, catalyst_signal, regime, rotation, basis_ticker, info["direction"])
        result.at[idx, "Catalyst Grade"] = catalyst_signal["label"]
        result.at[idx, "Buy Score"] = score["score"]
        result.at[idx, "Entry Quality"] = score["entry_quality"]
        result.at[idx, "Setup"] = score["label"]
        result.at[idx, "Score Basis"] = basis_ticker if info["is_leveraged"] else ticker
        plan = entry_plan(basis_metrics, score, info["direction"])
        if plan:
            result.at[idx, "Entry Zone"] = f"${plan['zone_low']:.2f}–${plan['zone_high']:.2f}"
            result.at[idx, "Breakout Trigger"] = plan["breakout_trigger"]
            result.at[idx, "Max Chase"] = plan["max_chase"]
            result.at[idx, "Stop Ref"] = plan["stop_ref"]
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
    "Technology": ["XLK", "VGT"],
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


with st.sidebar:
    LIVE_MODE = st.toggle(tr("盘中实时模式", "Intraday Live Mode"), value=True)
    STRICT_FRESHNESS = st.toggle(tr("严格防旧数据", "Strict stale-data guard"), value=True)
    MAX_LIVE_AGE = st.select_slider(
        tr("允许最大延迟", "Maximum allowed delay"),
        options=[5, 10, 15, 20, 30],
        value=20,
        format_func=lambda x: f"{x} min"
    )
    st.caption(tr(
        "盘中如果最新行情超过允许延迟，严格模式会拒绝给 Buy Score，不会偷偷退回昨天数据。",
        "During the session, if the newest bar exceeds the allowed delay, strict mode refuses to score instead of silently falling back to old daily data."
    ))


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
                if LIVE_MODE and len(out) <= 25:
                    st.caption(tr("候选 Buy Score 会尝试用盘中 1 分钟数据重新覆盖当前价和入场质量。", "Candidate Buy Scores will attempt a 1-minute intraday overlay for current price and entry quality."))
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
    st.caption(tr(
        "数据政策：盘中严格模式下，行情超过允许延迟就不评分；休市后只使用最近完成交易日，并明确标注日期。",
        "Data policy: in strict mode, stale intraday data blocks scoring; after hours, only the latest completed session is used and explicitly dated."
    ))

    b1, b2 = st.columns([1,1])
    with b1:
        bs_ticker = st.text_input(tr("Ticker", "Ticker"), value="MRNX").upper().strip()
    with b2:
        bs_override = st.text_input(tr("底层资产手动覆盖（可选）", "Underlying override (optional)"), value="").upper().strip()

    rcol1, rcol2 = st.columns([3,1])
    with rcol1:
        run_buy_score = st.button(tr("计算 / 刷新 Buy Score", "Calculate / Refresh Buy Score"), type="primary", use_container_width=True)
    with rcol2:
        st.caption(tr("实时缓存约 45 秒", "Live cache ~45 sec"))
    if run_buy_score:
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
                    basis = basis_rows.iloc[0].to_dict()
                    live_meta = {"live": False}
                    freshness = {"ok": True, "market_open": False, "stale": False}
                    if LIVE_MODE:
                        basis, live_meta = live_overlay_metrics(basis, info["underlying"])
                        freshness = freshness_status(live_meta, MAX_LIVE_AGE)
                        if freshness["ok"]:
                            basis = recompute_momentum_score(basis)

                    if LIVE_MODE and STRICT_FRESHNESS and not freshness["ok"]:
                        st.error(tr(
                            f"拒绝评分：{info['underlying']} 的盘中数据过旧或缺失。{freshness.get('message','')} 请稍后刷新；本次不会使用旧日线数据代替实时行情。",
                            f"Score blocked: intraday data for {info['underlying']} is stale or missing. {freshness.get('message','')} Refresh later; this run will not substitute old daily data."
                        ))
                        st.stop()

                    catalyst_signal = resolve_catalyst(info["underlying"], basis, rotation_bs, regime_bs)
                    grade = catalyst_signal["label"]
                    reason = catalyst_signal["reason"]
                    news = catalyst_signal["news"]
                    score = buy_score(basis, catalyst_signal, regime_bs, rotation_bs, info["underlying"], info["direction"])
                    product_day = float(product_rows.iloc[0]["Day %"]) if not product_rows.empty else np.nan
                    verdict = leverage_verdict(score, info["leverage"], product_day) if info["is_leveraged"] else "—"

                    a,b,c,d = st.columns(4)
                    a.metric(tr("Buy Score" if info["direction"] == 1 else "Directional Score", "Buy Score" if info["direction"] == 1 else "Directional Score"), f"{score['score']:.0f}/100", delta=score["label"])
                    b.metric(tr("Entry Quality", "Entry Quality"), f"{score['entry_quality']:.0f}/100")
                    c.metric(
                        tr("Catalyst", "Catalyst"),
                        grade,
                        delta=f"{score['components']['Catalyst']:.1f}/20"
                    )
                    d.metric(tr("Momentum", "Momentum"), f"{float(basis['Momentum Score']):.0f}/100")

                    if LIVE_MODE:
                        if freshness.get("market_open") and freshness.get("ok"):
                            age = freshness.get("age_minutes")
                            st.success(tr(
                                f"盘中数据已验证 · {live_meta.get('timestamp','')} · 延迟约 {age:.1f} 分钟",
                                f"Intraday data verified · {live_meta.get('timestamp','')} · about {age:.1f} min old"
                            ))
                        elif not freshness.get("market_open"):
                            if live_meta.get("live"):
                                st.info(tr(
                                    f"当前为休市时段；显示最近交易数据 · {live_meta.get('timestamp','')}",
                                    f"Market is closed; showing latest traded data · {live_meta.get('timestamp','')}"
                                ))
                            else:
                                st.info(tr(
                                    "当前为休市时段；本次使用最近一个已完成交易日的日线数据，并不会标记为实时。",
                                    "Market is closed; using the latest completed daily session, explicitly not marked as live."
                                ))
                        else:
                            st.error(tr(
                                "盘中数据未通过新鲜度检查。",
                                "Intraday data failed the freshness check."
                            ))

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

                    q1,q2,q3,q4,q5 = st.columns(5)
                    q1.metric(tr("当前价", "Current Price"), f"${float(basis['Price']):.2f}")
                    q2.metric(tr("底层 1D", "Underlying 1D"), f"{float(basis['Day %']):+.2f}%")
                    q3.metric(tr("底层 5D", "Underlying 5D"), f"{float(basis['5D %']):+.2f}%")
                    q4.metric("RVOL", f"{float(basis['RVOL']):.2f}x" if not np.isnan(basis['RVOL']) else "N/A")
                    q5.metric("ATR %", f"{float(basis['ATR %']):.2f}%" if not np.isnan(basis['ATR %']) else "N/A")

                    plan = entry_plan(basis, score, info["direction"])
                    if plan:
                        st.markdown(tr("#### 参考入场计划", "#### Reference Entry Plan"))
                        st.caption(tr(
                            f"以下价格全部以评分底层 {info['underlying']} 为依据，不是保证成交或盈利的价格预测。",
                            f"All levels below are based on the scoring underlying {info['underlying']}; they are reference levels, not guaranteed fills or profit forecasts."
                        ))

                        e1,e2,e3,e4 = st.columns(4)
                        e1.metric(
                            tr("推荐回踩区间", "Preferred Pullback Zone"),
                            f"${plan['zone_low']:.2f} – ${plan['zone_high']:.2f}"
                        )
                        e2.metric(
                            tr("突破确认价", "Breakout Trigger"),
                            f"${plan['breakout_trigger']:.2f}"
                        )
                        e3.metric(
                            tr("最高可追参考", "Max Chase Reference"),
                            f"${plan['max_chase']:.2f}"
                        )
                        e4.metric(
                            tr("参考止损位", "Reference Stop"),
                            f"${plan['stop_ref']:.2f}"
                        )

                        if info["direction"] == 1:
                            st.write(tr(
                                f"计划类型：{plan['setup']}。优先等价格进入回踩区间；如果不给回踩，则至少等 {info['underlying']} 有效突破 ${plan['breakout_trigger']:.2f}。高于约 ${plan['max_chase']:.2f} 后，V6.7 会把它视为追价区，不因为 Momentum 高就自动追。",
                                f"Plan type: {plan['setup']}. Prefer a pullback into the zone; if no pullback occurs, wait for {info['underlying']} to clear about ${plan['breakout_trigger']:.2f}. Above roughly ${plan['max_chase']:.2f}, V6.7 treats the move as chase territory rather than buying solely because momentum is high."
                            ))
                        else:
                            st.write(tr(
                                f"这是反向/看空产品的底层触发逻辑。优先看 {info['underlying']} 的阻力区和向下突破 ${plan['breakout_trigger']:.2f}。",
                                f"This is the underlying trigger logic for an inverse/bearish product. Focus on resistance and a downside break below about ${plan['breakout_trigger']:.2f} in {info['underlying']}."
                            ))

                        if info["is_leveraged"]:
                            st.warning(tr(
                                f"{bs_ticker} 是杠杆产品：这些入场价是 {info['underlying']} 的触发条件，不建议把底层的价格区间机械换算成 {bs_ticker} 的价格。底层先触发，再看杠杆 ETF 当时的盘口。",
                                f"{bs_ticker} is leveraged: these levels are triggers on {info['underlying']}. Do not mechanically convert them into a fixed {bs_ticker} price; wait for the underlying trigger, then evaluate the leveraged ETF at that moment."
                            ))

                    st.markdown(tr(
                        "#### 催化剂 / 行业信号" if catalyst_signal.get("kind") == "ETF" else "#### 催化剂新闻",
                        "#### Catalyst / Industry Signal" if catalyst_signal.get("kind") == "ETF" else "#### Catalyst News"
                    ))
                    st.write(reason)
                    if not news and catalyst_signal.get("kind") != "ETF":
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
