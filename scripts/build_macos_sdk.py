#!/usr/bin/env python3
"""Build the official SDK locally. No sudo and no device access during build."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import sysconfig

COMMIT = "34d6c778e1134d8505adcd56bb57acbe7598a459"
VERSION = "2.58.4"
USB_COMMIT = "cec1e49eec481a6333c218726ecaba8e4f447a07"
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=Path, default=ROOT / "build" / "realsense-sdk")
    parser.add_argument("--proxy", help="Optional GitHub HTTP proxy")
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--usb-version", choices=("1.0.30", "1.0.27"), default="1.0.30")
    args = parser.parse_args()
    if sys.platform != "darwin" or platform.machine() != "arm64":
        parser.error("此脚本用于 Apple Silicon Mac")
    for tool in ("cmake", "git", "curl", "otool", "install_name_tool", "codesign"):
        if not shutil.which(tool):
            parser.error(f"缺少 {tool}；请先安装 Xcode Command Line Tools 和 CMake")
    cache = args.build_root.resolve()
    source, build, includes = (cache / p for p in ("source", "build", "usb_include"))
    usb_source, usb_build = cache / "libusb", cache / "libusb-build"
    includes.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if args.proxy:
        env.update(GIT_CONFIG_COUNT="1", GIT_CONFIG_KEY_0="http.proxy", GIT_CONFIG_VALUE_0=args.proxy)
    def run(argv):
        subprocess.run([str(a) for a in argv], check=True, env=env)
    if not source.exists():
        run(["git", "clone", "--depth", "1", "--branch", f"v{VERSION}",
             "https://github.com/realsenseai/librealsense.git", source])
    actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if actual != COMMIT:
        parser.error(f"源码版本不匹配：{actual}")
    patch = ROOT / "patches" / "realsense-usb-status.patch"
    applied = subprocess.run(["git", "-C", str(source), "apply", "--reverse", "--check", str(patch)],
                             capture_output=True).returncode == 0
    if not applied:
        run(["git", "-C", source, "apply", "--check", patch])
        run(["git", "-C", source, "apply", patch])
    if args.usb_version == "1.0.27":
        if not usb_source.exists():
            run(["git", "clone", "--depth", "1", "--branch", "v1.0.27-1",
                 "https://github.com/libusb/libusb-cmake.git", usb_source])
        actual_usb = subprocess.check_output(["git", "-C", str(usb_source), "rev-parse", "HEAD"], text=True).strip()
        if actual_usb != USB_COMMIT:
            parser.error(f"libusb 源码版本不匹配：{actual_usb}")
        run(["cmake", "-S", usb_source, "-B", usb_build, "-DCMAKE_BUILD_TYPE=Release",
             "-DLIBUSB_BUILD_SHARED_LIBS=ON", "-DLIBUSB_BUILD_EXAMPLES=OFF", "-DLIBUSB_BUILD_TESTING=OFF"])
        run(["cmake", "--build", usb_build, "-j", str(max(1, min(args.jobs, 12)))])
        library = usb_build / "libusb-1.0.dylib"
        shutil.copy2(usb_source / "libusb" / "libusb" / "libusb.h", includes / "libusb.h")
        shutil.copy2(usb_source / "libusb" / "COPYING", includes / "COPYING")
    else:
        import libusb_package
        library = libusb_package.get_library_path()
        if library is None or libusb_package.__version__ != "1.0.30.0":
            parser.error("请先安装本项目的 mac 扩展（libusb-package 1.0.30.0）")
        curl = ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "60"]
        if args.proxy:
            curl += ["--proxy", args.proxy]
        for remote, filename in (("libusb/libusb.h", "libusb.h"), ("COPYING", "COPYING")):
            run([*curl, f"https://raw.githubusercontent.com/libusb/libusb/v1.0.30/{remote}", "--output", includes / filename])
    run(["cmake", "-S", source, "-B", build, "-DCMAKE_BUILD_TYPE=Release",
         "-DBUILD_PYTHON_BINDINGS=ON", "-DBUILD_EXAMPLES=OFF", "-DBUILD_GRAPHICAL_EXAMPLES=OFF",
         "-DBUILD_TOOLS=OFF", "-DBUILD_UNIT_TESTS=OFF", "-DBUILD_WITH_DDS=OFF", "-DCHECK_FOR_UPDATES=OFF",
         "-DFORCE_RSUSB_BACKEND=ON", "-DUSE_EXTERNAL_USB=OFF", f"-DLIBUSB_LIB={library}",
         f"-DLIBUSB_INC={includes}", f"-DPYTHON_EXECUTABLE={sys.executable}",
         f"-DPython_EXECUTABLE={sys.executable}", f"-DPYTHON_INSTALL_DIR={cache / 'install'}"])
    run(["cmake", "--build", build, "--target", "pyrealsense2", "-j", str(max(1, min(args.jobs, 12)))])
    output = ROOT / "vendor" / "realsense_macos"
    output.mkdir(parents=True, exist_ok=True)
    extension = "pyrealsense2" + sysconfig.get_config_var("EXT_SUFFIX")
    files = {extension: build / "Release" / extension,
             "librealsense2.2.58.dylib": build / "Release" / f"librealsense2.{VERSION}.dylib",
             "libusb-1.0.dylib": library}
    for name, path in files.items():
        shutil.copy2(path, output / name)
    run(["install_name_tool", "-change", "@rpath/librealsense2.2.58.dylib",
         "@loader_path/librealsense2.2.58.dylib", output / extension])
    # Use the actual install name in the bundled libusb, not a machine-specific guess.
    usb_id = subprocess.check_output(["otool", "-D", str(library)], text=True).splitlines()[1].strip()
    run(["install_name_tool", "-change", usb_id, "@loader_path/libusb-1.0.dylib", output / "librealsense2.2.58.dylib"])
    for name in files:
        run(["codesign", "--force", "--sign", "-", output / name])
    shutil.copy2(source / "LICENSE", output / "LICENSE.librealsense")
    shutil.copy2(includes / "COPYING", output / "LICENSE.libusb")
    (output / "build-info.json").write_text(json.dumps({
        "sdk_version": VERSION, "source": "https://github.com/realsenseai/librealsense", "commit": COMMIT,
        "python": platform.python_version(), "architecture": platform.machine(),
        "libusb_version": args.usb_version,
        "libusb_source": "libusb-cmake v1.0.27-1 " + USB_COMMIT if args.usb_version == "1.0.27" else "libusb-package 1.0.30.0",
        "patches": [patch.name], "mac_usb_precapture": True, "mac_rgb_aux_polling_disabled": True,
        "binary_sha256": {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in files},
        "macos_build": platform.mac_ver()[0]}, indent=2) + "\n")
    print(f"SDK 已安装到 {output}。关闭旧取图进程后，重新运行启动RealSense彩色流.command。")


if __name__ == "__main__":
    main()
