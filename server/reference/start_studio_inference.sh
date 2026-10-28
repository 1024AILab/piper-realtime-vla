#!/usr/bin/env bash
set -euo pipefail
source /root/autodl-tmp/piper-training/inference_env.sh
cd /root/autodl-tmp/piper-training/openpi
exec python /root/autodl-tmp/piper-training/start_studio_inference.py
