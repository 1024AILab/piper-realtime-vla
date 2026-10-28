"""Prefer the locally built macOS SDK, with a normal package fallback."""
import importlib
from pathlib import Path
import sys
import sysconfig


def load_sdk():
    native = Path(__file__).resolve().parent.parent / "vendor" / "realsense_macos"
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if sys.platform == "darwin" and (native / f"pyrealsense2{suffix}").is_file():
        if str(native) not in sys.path:
            sys.path.insert(0, str(native))
    sdk = importlib.import_module("pyrealsense2")
    if sys.platform == "darwin":
        version = tuple(int(v) for v in getattr(sdk, "__version__", "0").split(".")[:3])
        if version < (2, 58, 4):
            raise RuntimeError("Mac 取图需要本项目编译的 RealSense SDK 2.58.4 或更新版本；请使用 Python 3.12，或运行 scripts/build_macos_sdk.py 重建绑定")
    return sdk
