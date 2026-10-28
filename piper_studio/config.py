from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class CameraConfig:
    backend: str = "sim"  # sim, opencv, realsense, realsense_bridge
    device: str = "0"     # OpenCV index/path or RealSense serial
    fps: int = 30


@dataclass
class Config:
    mode: str = "sim"  # sim, hardware_bus, software
    dataset_root: str = "datasets/piper_dataset"
    can_backend: str = "socketcan"  # Linux SocketCAN or cross-platform gs_usb
    can_bitrate: int = 1000000
    follower_port: str = "can0"
    leader_port: str = "can1"
    leader_source: str = "control"  # control for a teaching pendant; feedback for passive arm
    fps: int = 10
    control_hz: int = 50
    width: int = 640
    height: int = 480
    max_episode_s: int = 300
    max_joint_speed_deg_s: float = 15.0
    max_gripper_speed_mm_s: float = 30.0
    max_follow_error_deg: float = 20.0
    stale_after_s: float = 0.5
    camera_stale_after_s: float = 0.6
    hold_stationary_targets: bool = True  # Hardware capture only; never used for robot control.
    motion_mode: str = "mit"  # stay in high-follow mode; no reset/zeroing is sent
    speed_percent: int = 10
    gripper_max_um: int = 70000
    joint_min_deg: list[float] = field(default_factory=lambda: [-150, -10, -175, -110, -70, -110])
    joint_max_deg: list[float] = field(default_factory=lambda: [150, 185, 10, 110, 70, 135])
    image: CameraConfig = field(default_factory=CameraConfig)
    wrist_image: CameraConfig = field(default_factory=lambda: CameraConfig(device="1"))

    def validate(self) -> None:
        if self.mode not in {"sim", "hardware_bus", "software"}:
            raise ValueError("mode 必须是 sim / hardware_bus / software")
        if self.leader_source not in {"control", "feedback"}:
            raise ValueError("leader_source 必须是 control 或 feedback")
        if not isinstance(self.hold_stationary_targets, bool):
            raise ValueError("hold_stationary_targets 必须是 true / false")
        if self.can_backend not in {"socketcan", "gs_usb"} or self.can_bitrate != 1000000:
            raise ValueError("CAN 后端须为 socketcan / gs_usb，PiPER 使用 1 Mbps")
        if self.mode != "sim" and self.can_backend == "gs_usb":
            ports = [self.follower_port] + ([self.leader_port] if self.mode == "software" else [])
            if any(not str(p).isdigit() for p in ports):
                raise ValueError("gs_usb 的端口填设备索引字符串，例如 \"0\"，不能填 can0")
        if self.fps != 10 or (self.width, self.height) != (640, 480):
            raise ValueError("样例兼容模式固定为 10 FPS、640×480 RGB")
        if not 20 <= self.control_hz <= 200:
            raise ValueError("control_hz 必须在 20–200 之间")
        if not 0 < self.max_joint_speed_deg_s <= 45:
            raise ValueError("关节速度必须在 (0, 45] °/s")
        if not 0 < self.max_gripper_speed_mm_s <= 100:
            raise ValueError("夹爪速度必须在 (0, 100] mm/s")
        if not 1 <= self.max_episode_s <= 3600:
            raise ValueError("单条时长必须在 1–3600 秒")
        if len(self.joint_min_deg) != 6 or len(self.joint_max_deg) != 6:
            raise ValueError("必须配置 6 个关节的运动范围")
        if any(a >= b for a, b in zip(self.joint_min_deg, self.joint_max_deg)):
            raise ValueError("关节最小值必须小于最大值")
        if not all(math.isfinite(x) for x in [*self.joint_min_deg, *self.joint_max_deg]):
            raise ValueError("关节范围必须为有限数值")
        if not 0.05 <= self.stale_after_s <= 2 or not 0.05 <= self.camera_stale_after_s <= 2 or not 1 <= self.max_follow_error_deg <= 45:
            raise ValueError("数据超时或跟随误差阈值不合法")
        if self.motion_mode not in {"mit", "position"} or not 1 <= self.speed_percent <= 30:
            raise ValueError("运动模式或速度比例无效")
        if not 1 <= self.gripper_max_um <= 100000:
            raise ValueError("夹爪最大行程必须在 1–100000 µm")
        if self.mode == "software" and self.leader_port == self.follower_port:
            raise ValueError("软件遥操需要两个独立 CAN 接口")
        for cam in (self.image, self.wrist_image):
            if cam.backend not in {"sim", "opencv", "realsense", "realsense_bridge"}:
                raise ValueError("未知相机 backend")
            if cam.backend == "realsense_bridge" and (not cam.device.isdigit() or len(cam.device) > 32):
                raise ValueError("RealSense 取图程序需要实际的数字序列号")
            if self.mode != "sim" and cam.backend == "sim":
                raise ValueError("真机采集不能使用模拟相机")
            if not 10 <= cam.fps <= 60:
                raise ValueError("相机帧率须在 10–60 FPS")

    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        if path is None:
            c = cls()
        else:
            raw = json.loads(Path(path).read_text())
            raw["image"] = CameraConfig(**raw.get("image", {}))
            raw["wrist_image"] = CameraConfig(**raw.get("wrist_image", {}))
            c = cls(**raw)
        c.validate()
        return c

    def to_dict(self) -> dict:
        return asdict(self)
