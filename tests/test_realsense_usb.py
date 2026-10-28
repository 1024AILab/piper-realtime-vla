import os
from pathlib import Path

import pytest

from piper_studio.realsense_usb import CameraUSBCapture


@pytest.fixture
def fake_usb(monkeypatch):
    usb_core = pytest.importorskip("usb.core")
    import usb.util
    from usb.backend import libusb1
    operations = []
    class Device:
        def __init__(self, serial, fail=False):
            self.serial_number, self.fail = serial, fail
        def detach_kernel_driver(self, interface):
            assert interface == 0
            if self.fail:
                raise RuntimeError("capture failed")
            operations.append((self.serial_number, "capture"))
        def attach_kernel_driver(self, interface):
            assert interface == 0
            operations.append((self.serial_number, "restore"))
    devices = [Device("123"), Device("456"), Device("789")]
    read_text = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda p,*a,**kw:'{"mac_usb_precapture":true}' if p.name=='build-info.json' else read_text(p,*a,**kw))
    monkeypatch.setattr(os, "geteuid", lambda:0)
    def get_backend(find_library):
        assert find_library("usb").endswith("/vendor/realsense_macos/libusb-1.0.dylib")
        return "shared SDK libusb"
    monkeypatch.setattr(libusb1, "get_backend", get_backend)
    def find(**kwargs):
        assert kwargs == dict(find_all=True,idVendor=0x8086,idProduct=0x0B07,backend="shared SDK libusb")
        return devices
    monkeypatch.setattr(usb_core, "find", find)
    monkeypatch.setattr(usb.util, "dispose_resources", lambda d:operations.append((d.serial_number,"dispose")))
    monkeypatch.setenv("PIPER_RS_USB_PRECAPTURED", "previous")
    return devices, operations


def test_camera_capture_owns_only_explicit_devices_and_restores_once(fake_usb):
    devices, operations = fake_usb
    capture = CameraUSBCapture(["456","123"]).open()
    assert operations == [("456","capture"),("123","capture")]
    assert os.environ["PIPER_RS_USB_PRECAPTURED"] == "1"
    capture.close()
    capture.close()
    assert operations == [("456","capture"),("123","capture"),("123","restore"),("456","restore"),
                          ("123","dispose"),("456","dispose"),("789","dispose")]
    assert os.environ["PIPER_RS_USB_PRECAPTURED"] == "previous"


def test_missing_camera_cannot_trigger_partial_capture(fake_usb):
    _, operations = fake_usb
    with pytest.raises(RuntimeError, match="未唯一识别"):
        CameraUSBCapture(["123","999"]).open()
    assert all(op == "dispose" for _,op in operations)
    assert os.environ["PIPER_RS_USB_PRECAPTURED"] == "previous"


def test_second_camera_failure_restores_first_camera(fake_usb):
    devices, operations = fake_usb
    devices[1].fail = True
    with pytest.raises(RuntimeError, match="capture failed"):
        CameraUSBCapture(["123","456"]).open()
    assert operations[:2] == [("123","capture"),("123","restore")]
    assert os.environ["PIPER_RS_USB_PRECAPTURED"] == "previous"
