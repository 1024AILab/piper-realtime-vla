# Third-party notices

- RealSense SDK: `vendor/realsense_macos/LICENSE.librealsense` (Apache-2.0).
- libusb: `vendor/realsense_macos/LICENSE.libusb` (LGPL-2.1-or-later).
- The bundled macOS binaries, source commit and hashes are documented in
  `vendor/realsense_macos/build-info.json`. The local SDK patch and rebuilding
  script are in `patches/` and `scripts/build_macos_sdk.py`.
- PiPER CAN protocol references: https://github.com/agilexrobotics/piper_sdk
- Dataset-layout reference: https://huggingface.co/datasets/LeoApollo/piper_single
- OpenPI protocol / serving: https://github.com/Physical-Intelligence/openpi

Dependency packages retain their respective licenses. This repository does not
relicense third-party code, model weights or datasets.

Original capture and realtime modules were separated from a local Piper Studio
snapshot dated 2026-09-13. The public backup was prepared on 2026-10-28.
Private deployment endpoints, device identifiers, recordings and credentials
are excluded. The unchanged original source snapshot is retained locally.
