import socket
import struct
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from piper_studio.realsense_bridge import make_server, read_frame, resolve_serials


def test_realsense_asic_ids_resolve_to_sdk_serials_without_camera_swaps():
    devices = [{"serial": "SDK-A", "asic": "ASIC-A"},
               {"serial": "SDK-B", "asic": "ASIC-B"}]
    assert resolve_serials(devices, ["ASIC-B", "ASIC-A"]) == {
        "ASIC-B": "SDK-B", "ASIC-A": "SDK-A"}
    with pytest.raises(RuntimeError, match="同一台"):
        resolve_serials(devices, ["ASIC-A", "SDK-A"])
    with pytest.raises(RuntimeError, match="未唯一识别"):
        resolve_serials(devices, ["999"])


def test_mac_can_receives_wire_frame_without_usb_reset_or_can_transmission(monkeypatch):
    module = pytest.importorskip("piper_studio.mac_can")
    operations = []
    class USB:
        def is_kernel_driver_active(self, _):
            return False
        def ctrl_transfer(self, kind, request, *args):
            operations.append((kind, request))
            if kind == 0xC1 and request == 4:
                return struct.pack("<10I", 0xF3, 48000000, 1, 16, 1, 8, 4, 1, 1024, 1)
        def read(self, endpoint, length, timeout):
            # Real gs_usb packet: RX echo sentinel, CAN ID, DLC/channel/flags,
            # two big-endian joint values and a little-endian USB timestamp.
            return bytes.fromhex("ffffffffa502000008000000000004d2fffff62b40e20100")
        def reset(self):
            pytest.fail("USB reset is not needed for Mac CAN initialization")
        def detach_kernel_driver(self, _):
            pytest.fail("Must not seize a kernel-owned device")
        def write(self, *args):
            pytest.fail("Read-only connection must not transmit CAN commands")
    device = USB()
    monkeypatch.setattr(module, "usb_backend", lambda: None)
    monkeypatch.setattr(module.usb.core, "find", lambda **_: [device])
    monkeypatch.setattr(module.usb.util, "claim_interface", lambda *_: operations.append("claim"))
    monkeypatch.setattr(module.usb.util, "release_interface", lambda *_: operations.append("release"))
    monkeypatch.setattr(module.usb.util, "dispose_resources", lambda *_: operations.append("dispose"))
    bus = module.MacGsUsbBus()
    frame = bus.recv(0.1)
    assert frame.arbitration_id == 0x2A5 and frame.is_rx
    assert struct.unpack(">ii", frame.data) == (1234, -2517)
    bus.shutdown()
    bus.shutdown()
    assert operations.count("claim") == operations.count("release") == operations.count("dispose") == 1


def test_mac_can_permission_failure_does_not_detach_or_start(monkeypatch):
    module = pytest.importorskip("piper_studio.mac_can")
    disposed = []
    device = SimpleNamespace(is_kernel_driver_active=lambda _: True)
    monkeypatch.setattr(module, "usb_backend", lambda: None)
    monkeypatch.setattr(module.usb.core, "find", lambda **_: [device])
    monkeypatch.setattr(module.usb.util, "dispose_resources", lambda dev: disposed.append(dev))
    with pytest.raises(module.can.CanInitializationError, match="系统驱动占用"):
        module.MacGsUsbBus()
    assert disposed == [device]


def test_rgb_bridge_preserves_pixels_acquisition_time_and_serial_identity():
    import os
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2] = 15, 80, 220
    stamp = time.monotonic() - 2  # A stale source MUST remain stale at the client.
    source = SimpleNamespace(snapshot=lambda: (rgb, stamp, ""))
    with tempfile.TemporaryDirectory(prefix="ps-test-", dir="/tmp") as temp:
        path = Path(temp) / "rgb.sock"
        server = make_server(path, {"123456789": source}, os.getuid())
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            image, received_stamp = read_frame("123456789", str(path))
            np.testing.assert_array_equal(image, rgb)
            assert received_stamp == stamp
            assert time.monotonic() - received_stamp >= 2
            assert path.stat().st_mode & 0o777 == 0o600
            with pytest.raises(RuntimeError, match="未配置此"):
                read_frame("999999999", str(path))
            with pytest.raises(RuntimeError, match="已在运行"):
                make_server(path, {}, os.getuid())
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=2)


def test_bridge_disconnect_is_detected():
    from piper_studio.realsense_bridge import recv_exact
    left, right = socket.socketpair()
    try:
        right.sendall(b"short")
        right.close()
        with pytest.raises(ConnectionError, match="断开"):
            recv_exact(left, 20)
    finally:
        left.close()
        right.close()


def test_realsense_recovers_after_initial_frame_timeout(monkeypatch):
    from piper_studio.realsense_bridge import RGBSource
    from piper_studio import realsense_runtime
    calls = []
    pixels = np.full((480, 640, 3), 27, dtype=np.uint8)
    operations = []
    class Queue:
        def try_wait_for_frame(self, timeout_ms):
            calls.append(timeout_ms)
            time.sleep(0.01)
            if len(calls) == 1:
                return False, None
            return True, SimpleNamespace(get_data=lambda: pixels)
    profile = SimpleNamespace(stream_type=lambda:1, format=lambda:1, fps=lambda:30,
                              as_video_stream_profile=lambda: SimpleNamespace(width=lambda:640, height=lambda:480))
    sensor = SimpleNamespace(supports=lambda _:True, set_option=lambda *v: operations.append(v),
                             get_stream_profiles=lambda:[profile], open=lambda _:operations.append("open"),
                             start=lambda _:operations.append("start"), stop=lambda:operations.append("stop"),
                             close=lambda:operations.append("close"))
    sdk = SimpleNamespace(frame_queue=lambda _:Queue(), option=SimpleNamespace(global_time_enabled=10),
                          stream=SimpleNamespace(color=1), format=SimpleNamespace(rgb8=1))
    monkeypatch.setattr(realsense_runtime, "load_sdk", lambda: sdk)
    source = RGBSource("123", SimpleNamespace(first_color_sensor=lambda:sensor), 30)
    try:
        deadline = time.monotonic() + 1
        while not source.first_frame and time.monotonic() < deadline:
            time.sleep(0.01)
        frame, stamp, error = source.snapshot()
        np.testing.assert_array_equal(frame, pixels)
        assert stamp > 0 and error == "" and len(calls) >= 2
    finally:
        source.close()
    assert operations == [(10,0),"open","start","stop","close"]
