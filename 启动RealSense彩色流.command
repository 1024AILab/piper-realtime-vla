#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
: "${PIPER_SCENE_SERIAL:?Set PIPER_SCENE_SERIAL to the scene camera USB identifier}"
: "${PIPER_WRIST_SERIAL:?Set PIPER_WRIST_SERIAL to the wrist camera USB identifier}"
mkdir -p runtime
printf '%s\n' 'RGB capture only. Keep this terminal open; camera access may need administrator permission.'
sudo -- /usr/bin/env LIBUSB_DEBUG=1 "$PWD/.venv/bin/python" -u -m piper_studio.realsense_bridge \
  --serial "$PIPER_SCENE_SERIAL" --serial "$PIPER_WRIST_SERIAL" 2>&1 | tee runtime/realsense.log
