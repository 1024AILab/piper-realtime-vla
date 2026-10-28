#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec .venv/bin/python -m piper_studio.inference_app --sim --connect --port 7862 --open
