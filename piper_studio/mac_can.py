"""candleLight on macOS, without resetting or detaching the USB device.

gs-usb 0.3.1 resets USB in start(), which can bind a kernel driver on macOS
and then fail during detach. A CAN-controller mode reset is sufficient here;
it is a USB configuration request, never a PiPER reset or CAN motion command.
"""
from __future__ import annotations

import can
import usb.core
import usb.util
from can.interfaces.gs_usb import GsUsbBus
from gs_usb.constants import GS_CAN_MODE_HW_TIMESTAMP
from gs_usb.gs_usb import GsUsb
from gs_usb.gs_usb_structures import DeviceMode


def usb_backend():
    from usb.backend import libusb1
    try:
        import libusb_package
    except ImportError:
        backend = libusb1.get_backend()
    else:
        backend = libusb_package.get_libusb1_backend()
    if backend is None:
        raise RuntimeError("缺少 libusb，请安装本项目的 mac 扩展：pip install -e '.[mac]'")
    return backend


class MacGsUsbBus(GsUsbBus):
    """Reuse python-can framing, with explicit USB ownership and cleanup."""

    def __init__(self, channel=0, bitrate=1000000):
        # BusABC.__del__ can run even when initialization fails.
        self._is_shutdown = True
        self._claimed = False
        self._device = None
        devices = list(usb.core.find(find_all=True, custom_match=GsUsb.is_gs_usb_device,
                                     backend=usb_backend()))
        if not 0 <= int(channel) < len(devices):
            raise can.CanInitializationError(f"未找到 USB-CAN {channel}，共发现 {len(devices)} 个兼容设备")
        self._device = devices[int(channel)]
        self.gs_usb = GsUsb(self._device)
        self.channel_info = f"gs_usb:{channel}"
        self._can_protocol = can.CanProtocol.CAN_20
        try:
            if self._device.is_kernel_driver_active(0):
                raise can.CanInitializationError("USB-CAN 接口被系统驱动占用；请关闭其他 CAN 软件并重新插拔转接器")
            usb.util.claim_interface(self._device, 0)
            self._claimed = True
            # Stop only this adapter's CAN controller before changing bit timing.
            self._device.ctrl_transfer(0x41, 2, 0, 0, DeviceMode(0, 0).pack())
            capability = self.gs_usb.device_capability
            timing = can.BitTiming.from_sample_point(f_clock=capability.fclk_can,
                                                     bitrate=bitrate, sample_point=87.5)
            self.gs_usb.set_timing(prop_seg=1, phase_seg1=timing.tseg1 - 1,
                                   phase_seg2=timing.tseg2, sjw=timing.sjw, brp=timing.brp)
            flags = GS_CAN_MODE_HW_TIMESTAMP & capability.feature
            self.gs_usb.device_flags = flags
            self._device.ctrl_transfer(0x41, 2, 0, 0, DeviceMode(1, flags).pack())
            can.BusABC.__init__(self, channel=channel)
        except Exception:
            self._release()
            raise

    def _release(self):
        if self._device is None:
            return
        try:
            if self._claimed:
                self.gs_usb.stop()
                usb.util.release_interface(self._device, 0)
        finally:
            usb.util.dispose_resources(self._device)
            self._claimed = False
            self._device = None

    def shutdown(self):
        if self._is_shutdown:
            return
        can.BusABC.shutdown(self)
        self._release()
