# 部署指南 / Deployment Guide

这个目录已经可以直接作为 GitHub 仓库根目录使用。

## A. 最快：Streamlit Community Cloud

1. 在 GitHub 新建一个 repository。
2. 把本目录里的所有文件上传到 repository 根目录；`.streamlit/config.toml` 也要保留。
3. 打开 Streamlit Community Cloud，创建 app 并连接这个 GitHub repository。
4. Branch 选 `main`，Main file path 选 `app.py`。
5. Deploy。
6. 部署完成后会得到一个 `*.streamlit.app` 地址，直接发给朋友。

目前程序没有 API key，所以不需要 secrets。如果未来加入付费行情 API，请不要把 key 写进 GitHub；使用 Streamlit 的 Secrets 设置。

## B. Fly.io

### 推荐方法：让 Fly 生成自己的 `fly.toml`

1. 安装 Fly CLI (`flyctl`) 并登录。
2. 在这个目录打开终端。
3. 第一次：

```bash
fly auth login
fly launch --generate-name
```

项目里已经有 Dockerfile，Fly 会用它构建 Streamlit。确认 internal port 为 `8080`。

4. 后续更新：

```bash
fly deploy
```

5. 打开：

```bash
fly open
```

### 手动配置

如果想自己维护 Fly 配置：

1. 把 `fly.toml.example` 复制为 `fly.toml`。
2. 把 `app = "replace-with-your-unique-app-name"` 改成你的唯一 app 名。
3. 创建/关联 app 后运行 `fly deploy`。

默认配置使用 1GB shared CPU，并允许空闲时自动停止、访问时自动启动，比较适合作为朋友测试版。

## C. 任意 Docker 托管

构建：

```bash
docker build -t momentum-scanner .
```

本地运行：

```bash
docker run --rm -p 8080:8080 momentum-scanner
```

打开 `http://localhost:8080`。

## D. 不使用 Docker 的通用 Python 托管

Build command:

```bash
pip install -r requirements.txt
```

Start command:

```bash
./start.sh
```

`start.sh` 会读取平台提供的 `$PORT`；如果没有，则使用 8501。

## 给朋友测试时建议让他们重点反馈

- 首页是否一眼知道先看 Market / Froth / Rotation / Momentum / Buy Score 的顺序。
- Buy Score 是否会被误解为“上涨概率”。
- 杠杆 ETF 页面里，正股/底层资产评分是否足够明显。
- 哪些字段太多、哪些字段还缺。
- 手机界面是否需要进一步简化。

## 已知限制

- 行情和新闻目前依赖 Yahoo/yfinance；可能延迟、缺失或触发限流。
- 多人同时频繁全市场扫描时，免费数据源更容易不稳定。
- 这是研究/筛选工具，不是券商行情终端，也不是自动交易系统。
