#!/usr/bin/env bash

set -e
set -x

python -m uvicorn bunka_api.routes:app --host 0.0.0.0
