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
