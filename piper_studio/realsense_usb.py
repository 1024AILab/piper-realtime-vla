"""Hold the selected D435 USB captures for one RGB helper session on macOS.

Use the SAME libusb dylib as the SDK. SDK interface handles must have automatic
detach/attach disabled by the project patch while this guard is active. This
avoids macOS re-enumerating the whole camera when a temporary SDK handle closes.
There are no CAN imports, resets of other devices, or firmware writes here.
"""
from pathlib import Path
import json
import os


class CameraUSBCapture:
    def __init__(self, identifiers):
        self.devices = []
        self.discovered = []
        self.previous_mode = os.environ.get("PIPER_RS_USB_PRECAPTURED")
        self.mode_set = False
        self.identifiers = list(identifiers)

    def open(self):
        from usb.backend import libusb1
        import usb.core
        import usb.util

        native = Path(__file__).resolve().parents[1] / "vendor/realsense_macos"
        metadata = json.loads((native / "build-info.json").read_text())
        if not metadata.get("mac_usb_precapture"):
            raise RuntimeError("请先运行 scripts/build_macos_sdk.py 更新本机 SDK 的 USB 持续连接补丁")
        if os.geteuid() != 0:
            raise RuntimeError("Mac D435 取图需要管理员 USB 权限；请运行“启动RealSense彩色流.command”")
        backend = libusb1.get_backend(find_library=lambda _: str(native / "libusb-1.0.dylib"))
        if backend is None:
            raise RuntimeError("无法加载本机 RealSense 配套 libusb")
        try:
            # Resolve both identities BEFORE capturing either camera.
            found = {}
            self.discovered = list(usb.core.find(find_all=True, idVendor=0x8086, idProduct=0x0B07, backend=backend))
            for dev in self.discovered:
                identifier = str(dev.serial_number)
                if identifier in found:
                    raise RuntimeError(f"D435 USB 标识重复：{identifier}")
                found[identifier] = dev
            missing = [s for s in self.identifiers if s not in found]
            if missing or len(set(self.identifiers)) != len(self.identifiers):
                raise RuntimeError(f"未唯一识别指定 D435 USB ASIC 标识 {missing or self.identifiers}；当前为 {list(found)}。请重连两台相机 USB 后再启动")
            for identifier in self.identifiers:
                dev = found[identifier]
                # On macOS this captures the whole camera once. No interface is
                # claimed here; the SDK claims the required RGB interfaces.
                dev.detach_kernel_driver(0)
                self.devices.append(dev)
                print(f"D435 USB {identifier} 已保持独立 USB 连接", flush=True)
            os.environ["PIPER_RS_USB_PRECAPTURED"] = "1"
            self.mode_set = True
        except BaseException:
            self.close()
            raise
        return self

    def close(self):
        import usb.util
        if self.mode_set:
            if self.previous_mode is None:
                os.environ.pop("PIPER_RS_USB_PRECAPTURED", None)
            else:
                os.environ["PIPER_RS_USB_PRECAPTURED"] = self.previous_mode
            self.mode_set = False
        for dev in reversed(self.devices):
            try:
                dev.attach_kernel_driver(0)
            except Exception as exc:
                print(f"相机 USB 连接恢复提示：{exc}；如相机不可见，请拔插其 USB 线", flush=True)
        self.devices.clear()
        for dev in self.discovered:
            usb.util.dispose_resources(dev)
        self.discovered.clear()
