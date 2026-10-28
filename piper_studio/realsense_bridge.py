"""Local RGB-only helper for macOS RealSense USB permission requirements.

No CAN/robot imports or controls. Raw RGB and original monotonic acquisition
timestamps travel over a Unix socket accessible only to the launching user.
Run the helper with sudo only when the SDK requires it; keep the UI unprivileged.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import signal
import socket
import socketserver
import stat
import struct
import sys
import threading
import time

import numpy as np

HEADER = struct.Struct("!4sBdI")
MAGIC = b"PSR1"
RGB_BYTES = 480 * 640 * 3


def default_socket(uid=None):
    uid = os.getuid() if uid is None else uid
    return f"/tmp/piper-studio-{uid}/realsense.sock"


def recv_exact(sock, length):
    chunks = bytearray()
    while len(chunks) < length:
        part = sock.recv(length - len(chunks))
        if not part:
            raise ConnectionError("RealSense 取图程序已断开")
        chunks.extend(part)
    return bytes(chunks)


def read_frame(serial, socket_path="", timeout=1.0):
    if not str(serial).isdigit() or len(str(serial)) > 32:
        raise ValueError("RealSense 序列号须为数字，最长 32 位")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        sock.connect(socket_path or default_socket())
        sock.sendall(str(serial).encode("ascii") + b"\n")
        magic, status, stamp, size = HEADER.unpack(recv_exact(sock, HEADER.size))
        if magic != MAGIC or status not in (0, 1) or size > RGB_BYTES:
            raise RuntimeError("RealSense 取图协议不匹配")
        if status == 1:
            if size > 4096:
                raise RuntimeError("RealSense 错误响应过长")
            raise RuntimeError(recv_exact(sock, size).decode("utf-8", errors="replace"))
        if size != RGB_BYTES or not math.isfinite(stamp) or stamp <= 0 or stamp > time.monotonic() + 0.01:
            raise RuntimeError("RealSense 图像大小或采集时间无效")
        frame = np.frombuffer(recv_exact(sock, size), dtype=np.uint8).reshape(480, 640, 3).copy()
        return frame, stamp


class RGBSource:
    def __init__(self, serial, device, fps):
        from .realsense_runtime import load_sdk
        rs = load_sdk()
        self.serial = serial
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.frame = None
        self.stamp = 0.0
        self.error = "等待第一帧 RGB"
        self.first_frame = False
        # Use the already resolved device, avoiding pipeline device-hub retries
        # and opening depth sensors solely to obtain host/global timestamps.
        self.device = device
        self.sensor = device.first_color_sensor()
        if self.sensor.supports(rs.option.global_time_enabled):
            self.sensor.set_option(rs.option.global_time_enabled, 0)
        profiles = [p for p in self.sensor.get_stream_profiles()
                    if p.stream_type() == rs.stream.color and p.format() == rs.format.rgb8
                    and p.fps() == fps and p.as_video_stream_profile().width() == 640
                    and p.as_video_stream_profile().height() == 480]
        if not profiles:
            raise RuntimeError(f"D435 {serial} 没有 RGB8 640×480 @ {fps} 配置")
        self.queue = rs.frame_queue(1)
        self.sensor.open(profiles[0])
        try:
            self.sensor.start(self.queue)
        except BaseException:
            self.sensor.close()
            raise
        self.thread = threading.Thread(target=self._run, daemon=True, name="RGB-" + serial)
        self.thread.start()

    def _run(self):
        try:
            while not self.stop_event.is_set():
                ok, color = self.queue.try_wait_for_frame(1000)
                if not ok:
                    # A cold start can take more than a second; keep waiting.
                    # Preserve the last acquisition timestamp, so data capture
                    # remains blocked while the camera is actually stale.
                    with self.lock:
                        self.error = "RGB 暂无新帧，正在继续等待"
                    continue
                if not color:
                    continue
                frame = np.asanyarray(color.get_data())
                if frame.shape != (480, 640, 3):
                    raise RuntimeError(f"RGB 输出尺寸错误：{frame.shape}")
                with self.lock:
                    self.frame = frame.copy()
                    self.stamp = time.monotonic()
                    self.error = ""
                if not self.first_frame:
                    print(f"D435 {self.serial} 首帧已收到：RGB8 640×480", flush=True)
                    self.first_frame = True
        except Exception as exc:
            with self.lock:
                self.error = str(exc)

    def snapshot(self):
        with self.lock:
            return self.frame, self.stamp, self.error

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=1.5)
        try:
            self.sensor.stop()
        finally:
            self.sensor.close()


class FrameHandler(socketserver.StreamRequestHandler):
    def handle(self):
        self.connection.settimeout(1)
        try:
            request = self.rfile.readline(34)
            if not request.endswith(b"\n"):
                return
            serial = request[:-1].decode("ascii")
            source = self.server.sources.get(serial)
            if source is None:
                raise RuntimeError("未配置此 RealSense 序列号")
            frame, stamp, error = source.snapshot()
            if error or frame is None:
                raise RuntimeError(error or "等待 RGB 图像")
            payload = frame.tobytes()
            self.wfile.write(HEADER.pack(MAGIC, 0, stamp, len(payload)) + payload)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass
        except Exception as exc:
            message = str(exc).encode("utf-8")[:4096]
            try:
                self.wfile.write(HEADER.pack(MAGIC, 1, 0, len(message)) + message)
            except OSError:
                pass


class FrameServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    block_on_close = False


def resolve_serials(devices, identifiers):
    """macOS USB serials are D435 ASIC IDs, not SDK enable_device serials."""
    resolved = {}
    for identifier in identifiers:
        matches = [d for d in devices if identifier in (d["serial"], d.get("asic"))]
        if len(matches) != 1:
            raise RuntimeError(f"相机 {identifier} 未唯一识别；SDK 当前识别到 {devices}")
        resolved[identifier] = matches[0]["serial"]
    if len(set(resolved.values())) != len(identifiers):
        raise RuntimeError("两个相机标识指向同一台设备")
    return resolved


def make_server(path, sources, uid):
    """Do not overwrite arbitrary files or use a shared/writable directory."""
    path = Path(path)
    if not path.parent.exists():
        path.parent.mkdir(mode=0o700)
        if os.geteuid() == 0:
            os.chown(path.parent, uid, -1)
    parent = path.parent.lstat()
    if not stat.S_ISDIR(parent.st_mode) or parent.st_uid != uid or parent.st_mode & 0o077:
        raise RuntimeError("取图 socket 目录必须属于当前用户，权限须为 0700")
    if path.exists() or path.is_symlink():
        existing = path.lstat()
        if not stat.S_ISSOCK(existing.st_mode) or existing.st_uid != uid:
            raise RuntimeError("取图 socket 路径已被其他文件占用")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.3)
            try:
                probe.connect(str(path))
            except ConnectionRefusedError:
                path.unlink()
            else:
                raise RuntimeError("RealSense 取图程序已在运行")
    server = FrameServer(str(path), FrameHandler)
    server.sources = sources
    os.chmod(path, 0o600)
    if os.geteuid() == 0:
        os.chown(path, uid, -1)
    return server


def main():
    parser = argparse.ArgumentParser(description="只读取指定 RealSense 的 RGB；不连接机械臂，不监听网络端口")
    parser.add_argument("--serial", action="append", required=True, help="可重复两次；明确指定 SDK 序列号或 USB ASIC 标识")
    parser.add_argument("--fps", type=int, choices=(15, 30, 60), default=30)
    args = parser.parse_args()
    if len(args.serial) != 2 or len(set(args.serial)) != 2 or any(not s.isdigit() or len(s) > 32 for s in args.serial):
        parser.error("请提供两台不同相机的数字序列号")
    uid = int(os.environ.get("SUDO_UID", os.getuid()))
    path = default_socket(uid)
    sources = {}
    server = None
    capture = None
    context = device = source = None
    device_handles = {}
    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    try:
        from .realsense_runtime import load_sdk
        rs = load_sdk()
        print(f"RealSense SDK {getattr(rs, '__version__', 'unknown')}", flush=True)
        build_info = Path(__file__).resolve().parents[1] / "vendor/realsense_macos/build-info.json"
        if build_info.is_file():
            print(f"libusb {json.loads(build_info.read_text()).get('libusb_version', 'unknown')} · 用户 ID {os.geteuid()} · 独立 RGB 传感器取图", flush=True)
        rs.log_to_console(rs.log_severity.warn)
        if sys.platform == "darwin":
            from .realsense_usb import CameraUSBCapture
            capture = CameraUSBCapture(args.serial).open()
        context = rs.context()
        # SDK initialization can fail or crash on macOS. It lives only in this
        # helper process, so the platform can continue displaying robot feedback.
        devices = []
        for device in context.query_devices():
            serial = device.get_info(rs.camera_info.serial_number)
            device_handles[serial] = device
            devices.append({"serial": serial,
                            "asic": device.get_info(rs.camera_info.asic_serial_number)
                            if device.supports(rs.camera_info.asic_serial_number) else None})
        serials = resolve_serials(devices, args.serial)
        for identifier, serial in serials.items():
            print(f"正在打开 D435 {serial}（USB 标识 {identifier}）· RGB8 640×480 @ {args.fps} FPS", flush=True)
            source = RGBSource(serial, device_handles[serial], args.fps)
            sources[identifier] = source
            deadline = time.monotonic() + 10
            while not source.first_frame and source.thread.is_alive() and time.monotonic() < deadline:
                time.sleep(0.1)
            if not source.first_frame:
                raise RuntimeError(f"D435 {serial} 在 10 秒内未收到 RGB，已停止本次启动；请检查前面的 USB 错误日志")
        server = make_server(path, sources, uid)
        print(f"两路真实 RGB 图像已就绪：{path}\n返回网页即可查看。按一次 Ctrl+C 停止取图，等待关闭完成。", flush=True)
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"RealSense 启动失败：{exc}", flush=True)
        raise SystemExit(1)
    finally:
        # A second Ctrl+C during native USB teardown can interrupt joins and
        # leave SDK threads touching already-destroyed objects.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        print("正在关闭相机取图…", flush=True)
        for source in sources.values():
            source.stop_event.set()
        if server:
            server.server_close()
            Path(path).unlink(missing_ok=True)
        for source in sources.values():
            try:
                source.close()
            except Exception as exc:
                print(f"关闭相机提示：{exc}", flush=True)
        # Release SDK device/context references BEFORE returning the whole USB
        # cameras to the OS. In particular keep guard handles alive throughout
        # native SDK worker and device-watcher shutdown.
        sources.clear()
        device_handles.clear()
        source = device = context = None
        gc.collect()
        if capture:
            capture.close()


if __name__ == "__main__":
    main()
