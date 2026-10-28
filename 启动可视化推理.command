#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec .venv/bin/python -m piper_studio.inference_app --config configs/mac_inference.local.json --connect --open
