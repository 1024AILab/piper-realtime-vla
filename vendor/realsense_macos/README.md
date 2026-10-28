# 本机 RealSense SDK

本目录中的 SDK 来自 RealSense 官方 `v2.58.4` 源码，提交 `34d6c778e1134d8505adcd56bb57acbe7598a459`。
包含上游 [macOS libusb 崩溃修复](https://github.com/realsenseai/librealsense/pull/14454)。
本机二进制用于 **macOS 26.5.2 / Apple Silicon / CPython 3.12**，其他 Python 或系统版本需要重新编译。

```bash
.venv/bin/python scripts/build_macos_sdk.py
```

需要 Xcode Command Line Tools、CMake 和 `.[mac]` 依赖。编译不使用 sudo，不打开设备。
源码与构建中间文件保留在 `build/realsense-sdk`。需要 GitHub 代理时可传 `--proxy http://127.0.0.1:端口`。

libusb 1.0.30 来自 `libusb-package 1.0.30.0`，以独立动态库连接。重建脚本也提供 `--usb-version 1.0.27` 以便复现实验；该旧版在本机试验中出现接口占用及初始化失败，当前不作为默认库。
项目补丁 `patches/realsense-usb-status.patch` 包含：真实 USB 错误返回、构造失败释放 handle、设备 watcher 先停止再销毁回调，以及 Mac 相机已经被辅助进程持有时关闭 SDK 的逐接口自动 detach/attach。最后一项仅在辅助进程建立相机 USB 会话后生效，通过进程内环境变量 `PIPER_RS_USB_PRECAPTURED=1` 启用；不要单独在终端设置这个变量。

辅助进程通过同一份 libusb 动态库取得并持有指定 D435 的 USB 使用权，待所有 SDK 设备和线程关闭后归还。完整版本记录见 `build-info.json`，`mac_usb_precapture` 标识修复能力。USB-CAN 使用自己独立的 libusb 副本，不受相机库替换影响。

Mac RGB 会话还会跳过 SDK 硬件时钟轮询和仅用于调试日志的 UVC 状态中断（`mac_rgb_aux_polling_disabled`）。关闭传感器的全局时间戳选项本身不会停止 SDK 的硬件时钟线程；本机曾在其反复打开/关闭深度接口时出现原生请求取消崩溃。辅助进程使用主机单调时钟标记原始取帧时间，直接 RGB 流不依赖这两个后台功能。常规 SDK 模式和其他操作系统不启用这些会话专用分支。
源代码：[RealSense](https://github.com/realsenseai/librealsense/tree/v2.58.4)、[libusb](https://github.com/libusb/libusb/tree/v1.0.30)、[备选 libusb 构建源码](https://github.com/libusb/libusb-cmake/tree/v1.0.27-1)。许可见同目录的 `LICENSE.*`。
