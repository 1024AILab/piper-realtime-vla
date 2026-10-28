from __future__ import annotations

import math
import logging
import os
import struct
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from filelock import FileLock, Timeout as FileLockTimeout

import numpy as np
from PIL import Image, ImageDraw

from .config import CameraConfig, Config
from .can_health import CanHealth


@dataclass
class ArmSample:
    action: np.ndarray | None
    follower: np.ndarray | None
    action_age: float
    feedback_age: float
    enabled: bool
    error: str = ""
    action_frame_ages: dict[str, float | None] = field(default_factory=dict)
    partial_action: np.ndarray | None = None


class CanFrames:
    """Track freshness of EACH of the three joint frames and the gripper frame.

    Offset CAN IDs are not supported. Local TX frames do not count as external
    leader commands, so replay can detect an accidentally reconnected leader.
    """
    JOINT_CONTROL = (0x155, 0x156, 0x157, 0x159)
    JOINT_FEEDBACK = (0x2A5, 0x2A6, 0x2A7, 0x2A8)
    MOTOR_STATUS = tuple(range(0x261, 0x267))
    STATUS = (0x2A1, *MOTOR_STATUS)
    EXTERNAL_CONTROL = (0x150, 0x151, *range(0x152, 0x160), 0x471)

    def __init__(self):
        self.lock = threading.Lock()
        self.frames = {}
        self.seen = {}
        self.external_control_at = 0.0

    def feed(self, arbitration_id, data, is_rx=True, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            self.seen[arbitration_id] = now
        if arbitration_id not in (*self.JOINT_CONTROL, *self.JOINT_FEEDBACK, *self.STATUS, *self.EXTERNAL_CONTROL) or len(data) != 8:
            return
        with self.lock:
            self.frames[arbitration_id] = (bytes(data), now)
            if is_rx and arbitration_id in self.EXTERNAL_CONTROL:
                self.external_control_at = now

    def recent_ids(self, within_s=1.0):
        now = time.monotonic()
        with self.lock:
            return sorted(identifier for identifier, stamp in self.seen.items() if now - stamp <= within_s)

    def read(self, control=False, now=None):
        values, ages = self.read_parts(control, now)
        if any(age is None for age in ages.values()):
            return None, float("inf")
        return values, max(ages.values())

    def read_parts(self, control=False, now=None):
        """Keep per-frame age and partial values for diagnostics, never fill missing targets."""
        now = time.monotonic() if now is None else now
        ids = self.JOINT_CONTROL if control else self.JOINT_FEEDBACK
        with self.lock:
            frames = [self.frames.get(i) for i in ids]
        joints = np.full(7, np.nan, dtype=np.float32)
        ages = {}
        for index, (identifier, frame) in enumerate(zip(ids, frames)):
            ages[hex(identifier)] = max(0.0, now - frame[1]) if frame else None
            if frame is not None:
                if index < 3:
                    joints[index * 2:index * 2 + 2] = struct.unpack(">ii", frame[0])
                else:
                    joints[6] = struct.unpack(">i", frame[0][:4])[0]
        return joints, ages

    def status(self, stale_after_s, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            frame = self.frames.get(0x2A1)
            motors = [self.frames.get(i) for i in self.MOTOR_STATUS]
        if not frame or now - frame[1] > stale_after_s:
            return False, "机械臂状态帧 0x2A1 缺失或超时"
        data = frame[0]
        error = struct.unpack(">H", data[6:8])[0]
        if error or data[1] not in (0, 0x0B, 0x0C, 0x0D):
            return False, f"Piper status={data[1]:#x}, err_code={error:#x}"
        if any(m is None or now - m[1] > stale_after_s for m in motors):
            return False, "电机状态帧 0x261–0x266 缺失或超时"
        # Bit 6 is enable; bits 0–5 are fault flags; bit 7 is motor stall.
        if any(m[0][5] & 0xBF for m in motors):
            return False, "电机状态报告电压、温度、碰撞或驱动器故障"
        return all(bool(m[0][5] & 0x40) for m in motors), ""


class _CanObserver:
    def __init__(self, port, backend="socketcan", bitrate=1000000):
        import can
        if backend == "gs_usb":
            if sys.platform == "darwin":
                from .mac_can import MacGsUsbBus
                self.bus = MacGsUsbBus(channel=int(port), bitrate=bitrate)
            else:
                self.bus = can.Bus(interface="gs_usb", channel=int(port), bitrate=bitrate)
        else:
            self.bus = can.Bus(interface="socketcan", channel=port, receive_own_messages=False)
        self.frames = CanFrames()
        self.health = CanHealth()
        self.port = str(port)
        self.stop_event = threading.Event()
        self.transport_error = ""
        self.thread = threading.Thread(target=self._run, daemon=True, name=f"CAN-observer-{port}")
        self.thread.start()

    def _run(self):
        try:
            while not self.stop_event.is_set():
                msg = self.bus.recv(timeout=0.1)
                if msg is not None:
                    if msg.is_error_frame:
                        event = self.health.feed_error(msg.arbitration_id, msg.data)
                        if event:
                            logging.getLogger(__name__).warning("CAN[%s] %s mask=%s data=%s",
                                self.port, event["message"], event["mask"], event["data"])
                    elif not msg.is_remote_frame and not msg.is_extended_id:
                        self.frames.feed(msg.arbitration_id, msg.data, msg.is_rx)
                        if msg.is_rx:
                            self.health.feed_valid()
        except Exception as exc:
            self.transport_error = f"CAN 接收线程已停止：{exc}"

    @property
    def error(self):
        return self.transport_error or self.health.snapshot()["error"]

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=1)
        self.bus.shutdown()

    def send(self, arbitration_id, payload):
        import can
        self.bus.send(can.Message(arbitration_id=arbitration_id, data=payload, is_extended_id=False), timeout=0.1)


def command_frames(raw, config):
    """Piper V2 position commands; wire values cross-checked against piper-sdk 0.6.2."""
    from .motion import validate_targets
    q = [int(round(float(x))) for x in validate_targets([raw], config)[0]]
    mit = 0xAD if config.motion_mode == "mit" else 0x00
    return [(0x151, bytes([1, 1, config.speed_percent, mit, 0, 0, 0, 0])),
            *[(0x155 + i, struct.pack(">ii", *q[i * 2:i * 2 + 2])) for i in range(3)],
            (0x159, struct.pack(">iHBB", q[6], 1000, 1, 0))]


class PiperHardware:
    def __init__(self, config: Config):
        self.config = config
        self.follower_observer = None
        self.leader_observer = None
        self.connected_at = 0.0
        self._leases = []

    def connect(self):
        if self.config.can_backend == "socketcan" and sys.platform != "linux":
            raise RuntimeError("SocketCAN 仅支持 Linux。Mac 请使用 gs_usb 和兼容的 USB-CAN 转接器")
        try:
            directory = Path(tempfile.gettempdir()) / f'piper-studio-can-{os.getuid()}'
            directory.mkdir(mode=0o700, exist_ok=True)
            ports = {self.config.follower_port}
            if self.config.mode == 'software':
                ports.add(self.config.leader_port)
            for port in sorted(ports):
                # Same lock in capture and inference; never let a second process
                # reset the adapter's controller while the first owns the bus.
                name = str(port).replace('/', '_')
                lease = FileLock(directory / f'{self.config.can_backend}-{name}.lock', thread_local=False)
                try:
                    lease.acquire(timeout=0)
                except FileLockTimeout as exc:
                    raise RuntimeError('USB-CAN 正由另一个 Piper Studio 进程使用；请先在另一页面断开设备') from exc
                self._leases.append(lease)
            self.follower_observer = _CanObserver(self.config.follower_port, self.config.can_backend, self.config.can_bitrate)
            if self.config.mode == "software":
                self.leader_observer = _CanObserver(self.config.leader_port, self.config.can_backend, self.config.can_bitrate)
            else:
                self.leader_observer = self.follower_observer
            self.connected_at = time.monotonic()
        except Exception:
            self.close()
            raise

    def read(self) -> ArmSample:
        partial, ages = self.leader_observer.frames.read_parts(control=self.config.leader_source == "control")
        complete = all(age is not None for age in ages.values())
        action = partial.copy() if complete else None
        action_age = max(ages.values()) if complete else float("inf")
        feedback, feedback_age = self.follower_observer.frames.read(control=False)
        enabled, status_error = self.follower_observer.frames.status(self.config.stale_after_s)
        error = self.follower_observer.error or self.leader_observer.error or status_error
        return ArmSample(action, feedback, action_age, feedback_age, enabled, error, ages, partial)

    def external_control_recent(self, quiet_s=1.0) -> bool:
        return time.monotonic() - max(self.connected_at, self.follower_observer.frames.external_control_at) < quiet_s

    def diagnostics(self):
        observers = [("主从 CAN" if self.config.mode == "hardware_bus" else "从臂 CAN", self.follower_observer)]
        if self.leader_observer is not self.follower_observer:
            observers.append(("主臂 CAN", self.leader_observer))
        return [{"name": name, "port": observer.port, **observer.health.snapshot(),
                 "transport_error": observer.transport_error}
                for name, observer in observers if observer is not None]

    def prepare(self):
        """Enable only after explicit operator action; never reset or disable leader."""
        if self.external_control_recent():
            raise RuntimeError("从臂总线上仍有外部主臂控制帧，或静默观察不足 1 秒；请断开主臂后重试")
        current = self.read()
        if current.follower is None or current.feedback_age > self.config.stale_after_s or current.error:
            raise RuntimeError("从臂反馈不新鲜或存在故障，不能准备执行")
        self.follower_observer.send(0x471, bytes([7, 2, 0, 0, 0, 0, 0, 0]))
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if self.read().enabled:
                # Capture current pose immediately before holding, not the pose before enable.
                current = self.read()
                if current.follower is None or current.feedback_age > self.config.stale_after_s or current.error or self.external_control_recent():
                    raise RuntimeError("使能后反馈失效或出现外部控制帧；没有发送位置目标")
                self.send(current.follower)
                return
            time.sleep(0.05)
        raise RuntimeError("从臂使能超时；未自动 reset，请检查固件、模式和状态")

    def request_enable(self):
        """One enable request; inference controller supervises freshness/cancellation."""
        if self.external_control_recent():
            raise RuntimeError("检测到外部 CAN 控制，不能使能")
        self.follower_observer.send(0x471, bytes([7, 2, 0, 0, 0, 0, 0, 0]))

    def send(self, raw):
        if raw is None:
            raise RuntimeError("无有效目标")
        frames = command_frames(raw, self.config)
        for identifier, payload in frames:
            self.follower_observer.send(identifier, payload)
        sent = [v for _, payload in frames[1:4] for v in struct.unpack(">ii", payload)]
        sent.append(struct.unpack(">i", frames[-1][1][:4])[0])
        return np.asarray(sent, dtype=np.float32)

    def hold(self):
        sample = self.read()
        if sample.follower is not None and sample.feedback_age < self.config.stale_after_s and not sample.error and not self.external_control_recent():
            self.send(sample.follower)

    def close(self):
        # Disconnect communications only. Disabling torque can make the arm fall.
        observers = {o for o in (self.follower_observer, self.leader_observer) if o is not None}
        try:
            for observer in observers:
                observer.close()
        finally:
            self.follower_observer = self.leader_observer = None
            for lease in self._leases:
                lease.release()
            self._leases.clear()


class SimRobot:
    def __init__(self, config):
        self.config = config
        self.started = time.monotonic()
        self.q = np.asarray([0, 30000, -45000, 0, 25000, 0, 35000], dtype=np.float32)
        self.target = self.q.copy()
        self.demo = True
        self.enabled = True
        self.error = ""
        self.send_count = 0

    def connect(self):
        self.started = time.monotonic()

    def read(self):
        t = time.monotonic() - self.started
        if self.demo:
            base = np.asarray([0, 42000, -55000, 0, 24000, 0, 35000])
            amp = np.asarray([15000, 12000, 10000, 9000, 8000, 12000, 22000])
            self.target = (base + amp * np.sin(t * 0.55 + np.arange(7) * 0.6)).astype(np.float32)
        self.q += (self.target - self.q) * 0.3
        return ArmSample(self.target.copy(), self.q.copy(), 0.0, 0.0, self.enabled, self.error)

    def external_control_recent(self, quiet_s=1.0):
        return False

    def prepare(self):
        self.demo = False

    def send(self, raw):
        self.target = np.asarray(raw, dtype=np.float32).copy()
        self.send_count += 1
        return self.target.copy()

    def hold(self):
        self.target = self.q.copy()

    def close(self):
        pass


def simulation_image(t: float, wrist=False) -> np.ndarray:
    """A labelled diagnostic animation, not simulated camera physics."""
    im = Image.new("RGB", (640, 480), "#111d2c")
    d = ImageDraw.Draw(im)
    for x in range(0, 641, 40):
        d.line((x, 0, x, 480), fill="#1c2c3e", width=1)
    for y in range(0, 481, 40):
        d.line((0, y, 640, y), fill="#1c2c3e", width=1)
    d.rounded_rectangle((38, 55, 602, 430), radius=8, outline="#42576e", width=2)
    d.rectangle((440, 280, 550, 370), outline="#67e8bd", width=4)
    d.text((453, 378), "TARGET", fill="#67e8bd")
    if wrist:
        x, y = 320 + 70 * math.sin(t * 0.55), 260 + 30 * math.cos(t * 0.55)
        d.rectangle((x - 27, y - 27, x + 27, y + 27), fill="#f49b59")
        d.line((280, 125, 280, 220), fill="#b9c5d3", width=14)
        d.line((360, 125, 360, 220), fill="#b9c5d3", width=14)
        d.line((295, 240, 345, 240), fill="#65d7ee", width=1)
        d.line((320, 215, 320, 265), fill="#65d7ee", width=1)
    else:
        angle = t * 0.55
        elbow = (230 + 45 * math.sin(angle), 250 + 30 * math.cos(angle))
        tip = (370 + 100 * math.sin(angle), 190 + 50 * math.cos(angle))
        d.line([(90, 365), elbow, tip], fill="#b9c5d3", width=24)
        for x, y in ((90, 365), elbow, tip):
            d.ellipse((x - 18, y - 18, x + 18, y + 18), fill="#253e54", outline="#a2b7c8", width=3)
        d.rectangle((tip[0] - 25, tip[1] + 20, tip[0] + 25, tip[1] + 42), outline="#67e8bd", width=4)
        d.rectangle((320, 315, 365, 360), fill="#f49b59")
    d.text((20, 18), "SIMULATED CAMERA / " + ("WRIST" if wrist else "FRONT"), fill="#9aafc3")
    d.text((20, 454), f"640 x 480 RGB   {t:07.2f}s", fill="#91a9bf")
    return np.asarray(im)


class Camera:
    def __init__(self, config: CameraConfig, wrist=False):
        self.config = config
        self.wrist = wrist
        self.image = None
        self.stamp = 0.0
        self.error = ""
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.capture = None
        self.pipeline = None
        self.thread = None
        self.started = time.monotonic()

    def connect(self):
        if self.config.backend == "opencv":
            import cv2
            device = int(self.config.device) if self.config.device.isdigit() else self.config.device
            self.capture = cv2.VideoCapture(device, cv2.CAP_AVFOUNDATION) if sys.platform == "darwin" and isinstance(device, int) else cv2.VideoCapture(device)
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.capture.set(cv2.CAP_PROP_FPS, self.config.fps)
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not self.capture.isOpened():
                raise RuntimeError(f"无法打开相机 {device}")
        elif self.config.backend == "realsense":
            from .realsense_runtime import load_sdk
            rs = load_sdk()
            self.pipeline = rs.pipeline()
            cfg = rs.config()
            if not self.config.device or self.config.device.startswith("REPLACE"):
                raise ValueError("请填写实际 RealSense 序列号")
            cfg.enable_device(self.config.device)
            cfg.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, self.config.fps)
            self.pipeline.start(cfg)
        self.thread = threading.Thread(target=self._run, daemon=True, name="camera-" + self.config.device)
        self.thread.start()

    def _run(self):
        try:
            while not self.stop_event.is_set():
                acquired_at = None
                if self.config.backend == "sim":
                    frame = simulation_image(time.monotonic() - self.started, self.wrist)
                    self.stop_event.wait(1 / self.config.fps)
                elif self.config.backend == "opencv":
                    import cv2
                    ok, bgr = self.capture.read()
                    if not ok:
                        raise RuntimeError("相机断开或无法读取图像")
                    frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                elif self.config.backend == "realsense_bridge":
                    from .realsense_bridge import read_frame
                    try:
                        frame, acquired_at = read_frame(self.config.device)
                    except (OSError, RuntimeError) as exc:
                        if isinstance(exc, (FileNotFoundError, ConnectionRefusedError)):
                            self.error = "彩色取图程序未运行或启动失败；请查看取图终端，并启动“启动RealSense彩色流.command”"
                        else:
                            self.error = f"等待 RealSense 彩色取图程序：{exc}"
                        self.stop_event.wait(0.3)
                        continue
                    self.error = ""
                    self.stop_event.wait(1 / self.config.fps)
                else:
                    frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                    color = frames.get_color_frame()
                    if not color:
                        continue
                    frame = np.asanyarray(color.get_data())
                if frame.shape != (480, 640, 3):
                    raise RuntimeError(f"相机输出 {frame.shape}，要求 480×640×3；请调整设备配置")
                with self.lock:
                    self.image = frame.copy()
                    self.stamp = acquired_at if acquired_at is not None else time.monotonic()
        except Exception as exc:
            self.error = str(exc)

    def read(self):
        with self.lock:
            return self.image, time.monotonic() - self.stamp if self.stamp else float("inf"), self.stamp

    def close(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=1.5)
        if self.capture:
            self.capture.release()
        if self.pipeline:
            try:
                self.pipeline.stop()
            except RuntimeError:
                pass


def discover_cameras() -> dict:
    result = {"opencv": [], "realsense": []}
    try:
        # Isolate native SDK crashes (seen on macOS when USB access is denied).
        import subprocess
        import json
        code = "from piper_studio.realsense_runtime import load_sdk; import json; rs=load_sdk(); print(json.dumps([{'serial':d.get_info(rs.camera_info.serial_number),'name':d.get_info(rs.camera_info.name)} for d in rs.context().devices]))"
        run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=8)
        if run.returncode:
            raise RuntimeError(run.stderr.strip().splitlines()[-1] if run.stderr.strip() else f"SDK 退出码 {run.returncode}")
        result["realsense"] = json.loads(run.stdout)
    except Exception as exc:
        result["realsense_note"] = f"RealSense SDK 不可用：{exc}。Mac 可先检查相机能否作为普通 RGB 相机打开。"
    if sys.platform == "linux":
        from pathlib import Path
        result["opencv"] = [str(p) for p in Path("/dev").glob("video*")]
        result["can"] = [p.name for p in Path("/sys/class/net").iterdir() if p.name.startswith("can")]
    if sys.platform == "darwin":
        result["opencv_note"] = "Mac 使用 AVFoundation 相机索引（0、1…），具体对应关系请在实时画面核对；不要把内置摄像头当作腕部相机。"
    try:
        import usb.core
        from gs_usb.gs_usb import GsUsb
        from .mac_can import usb_backend
        devices = list(usb.core.find(find_all=True, custom_match=GsUsb.is_gs_usb_device, backend=usb_backend()))
        result["gs_usb"] = []
        for i, d in enumerate(devices):
            try:
                result["gs_usb"].append({"index": i, "usb_bus": d.bus, "usb_address": d.address,
                                         "serial": d.serial_number, "name": d.product})
            finally:
                import usb.util
                usb.util.dispose_resources(d)
    except Exception as exc:
        result["gs_usb_note"] = f"无法枚举 gs_usb 设备：{exc}。Mac 需安装本项目的 mac 扩展。"
    return result
