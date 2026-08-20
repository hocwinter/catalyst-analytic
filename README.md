# V6.7 — Freshness Guard

Critical data-safety change:
- Strict stale-data guard is ON by default.
- User-selectable maximum live delay: 5 / 10 / 15 / 20 / 30 minutes.
- During US extended trading hours (04:00–20:00 ET, weekdays), stale or missing intraday data BLOCKS Buy Score.
- The app no longer silently falls back to yesterday's daily bar during an active session.
- Outside trading hours, the latest completed session can be used, but it is explicitly labeled as non-live / latest traded data.
- Momentum candidate rows with stale live data are marked STALE DATA instead of receiving a misleading score.

This is designed specifically to prevent old prices from producing false Buy Scores.

# V6.6 Live — Intraday Buy Score

New:
- 1-minute intraday overlay for focused Buy Score checks.
- Current price, current-day %, day high/low, close location, breakout state and extension are recalculated intraday.
- Entry Quality and Entry Plan therefore react during the session instead of waiting for the daily close.
- Approximate same-time intraday RVOL uses 5-minute history when Yahoo returns enough data.
- Live timestamp is shown.
- Refresh button is built into Buy Score.
- If intraday data fails, the app falls back to daily data instead of crashing.
- Momentum candidate lists of 25 names or fewer can also receive live overlays.

Important:
Yahoo/yfinance is near-real-time research data, not exchange-direct or broker-grade market data. It may be delayed or rate-limited.

# V6.5 — ETF Catalyst Fix

New ETF-aware catalyst scoring.

Regular stocks:
- Company news / earnings / FDA / contracts / guidance remain the catalyst source.

Sector and industry ETFs:
- Catalyst score is now based on sector leadership, relative strength vs SPY,
  20-day breakout breadth, median RVOL, persistence/state, and ETF confirmation.
- No longer defaults to 4/20 just because the ETF has no company-style headline.

Broad-market ETFs:
- Catalyst uses market regime plus the ETF's own trend/breakout confirmation.

Leveraged ETFs:
- Still score the underlying/proxy first.
- If the underlying is an ETF (e.g. SOXL -> SMH), the new ETF catalyst model is used.

Also added Technology (XLK/VGT) to the rotation universe.

# V6.4 — Entry Plan Upgrade

New:
- Preferred pullback entry zone
- Breakout confirmation trigger
- Max-chase reference
- ATR-aware reference stop
- Entry levels are calculated on the underlying asset for leveraged ETFs
- Scanner candidate table can carry Entry Zone / Breakout Trigger / Max Chase / Stop Ref

The entry plan is volatility-aware and uses current price, ATR, 10-day support, 20-day high/low and breakout status. It is a research framework, not a guaranteed price forecast.


## V6.3 hotfix

- Fixed single-ticker yfinance parsing.
- Valid symbols such as `BE` no longer require an artificial second ticker or underlying override.
- Handles both flat and MultiIndex OHLCV column layouts returned by different yfinance versions.
- Leveraged ETF underlying mapping remains unchanged.

# Momentum + Buy Score + Froth + Rotation Scanner V6.2 Deploy

一个可直接部署分享的 Streamlit 研究工具。

## 功能

- 中文 / English 切换
- Daily Workflow
- Momentum Scanner
- Buy Score（与 Momentum Score 分开）
- 常见杠杆 ETF 自动映射到底层资产后评分
- Catalyst 检查
- Market Froth Gauge
- VIX + SPY/QQQ Market Regime
- Sector / Industry Rotation

## 本地 Windows

解压后双击 `run_windows.bat`。

## GitHub + Streamlit Cloud

把整个目录上传到 GitHub，Community Cloud 的 main file 指向 `app.py`。

## Fly.io

项目已包含：

- `Dockerfile`
- `.streamlit/config.toml`
- `.dockerignore`
- `fly.toml.example`

进入目录后运行：

```bash
fly auth login
fly launch --generate-name
```

以后更新：

```bash
fly deploy
```

详细步骤见 `DEPLOY.md`。

## 注意

Yahoo/yfinance 不是券商级实时数据源，可能出现延迟、缺失和限流。Buy Score 是 setup quality 评分，不是上涨概率。
