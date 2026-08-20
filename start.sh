#!/usr/bin/env sh
set -eu
exec streamlit run app.py \
  --server.address=0.0.0.0 \
  --server.port="${PORT:-8501}" \
  --server.headless=true
