#!/usr/bin/env bash
set -euo pipefail

python -m app.model_download

exec uvicorn app.main:app --host 0.0.0.0 --port 8000
