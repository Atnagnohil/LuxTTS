#!/bin/bash
set -a
source .env
set +a

.venv/bin/python3 -m uvicorn api.server:app --host 0.0.0.0 --port 8000
