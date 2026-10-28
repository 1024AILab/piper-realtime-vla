"""Position-only joint replay. All internal values use the reference's raw units."""
from __future__ import annotations

import numpy as np

from .config import Config


def validate_targets(actions, config: Config) -> np.ndarray:
    a = np.asarray(actions, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 7 or len(a) == 0 or not np.isfinite(a).all():
        raise ValueError("回放动作必须为有限的 N×7 原始单位数组")
    if np.any(a[:, :6] < np.asarray(config.joint_min_deg) * 1000) or np.any(a[:, :6] > np.asarray(config.joint_max_deg) * 1000):
        raise ValueError("轨迹超出配置的关节范围；请核实单位、标定和硬件限位")
    # Reference gripper feedback sometimes has small negative encoder noise.
    if np.any(a[:, 6] < -1000) or np.any(a[:, 6] > config.gripper_max_um + 1000):
        raise ValueError("夹爪轨迹超出允许行程")
    a = a.copy()
    a[:, 6] = np.clip(a[:, 6], 0, config.gripper_max_um)
    return a


def speed_limits(config):
    return np.asarray([config.max_joint_speed_deg_s * 1000] * 6 + [config.max_gripper_speed_mm_s * 1000])


class Trajectory:
    def __init__(self, current, actions, config: Config, speed=0.5):
        if not 0.1 <= speed <= 1:
            raise ValueError("真机回放倍率必须在 0.1–1.0")
        self.actions = validate_targets(actions, config)
        self.current = validate_targets([current], config)[0]
        self.limits = speed_limits(config)
        self.approach_s = max(1.0, float(np.max(np.abs(self.actions[0] - self.current) / self.limits)) * 1.875)
        peak = np.max(np.abs(np.diff(self.actions, axis=0)) * config.fps / self.limits) if len(self.actions) > 1 else 0
        self.speed = min(speed, 1 / max(peak, 1e-9))
        self.duration = self.approach_s + max(len(self.actions) - 1, 0) / config.fps / self.speed
        self.fps = config.fps

    def at(self, elapsed):
        if elapsed < self.approach_s:
            u = np.clip(elapsed / self.approach_s, 0, 1)
            blend = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5
            return self.current + (self.actions[0] - self.current) * blend, 0, False
        p = max(0, elapsed - self.approach_s) * self.fps * self.speed
        i = min(int(p), len(self.actions) - 1)
        j = min(i + 1, len(self.actions) - 1)
        q = self.actions[i] * (1 - min(p - i, 1)) + self.actions[j] * min(p - i, 1)
        return q, i, elapsed >= self.duration


def limited_step(current, target, dt, config):
    target = validate_targets([target], config)[0]
    return np.asarray(current) + np.clip(target - current, -speed_limits(config) * dt, speed_limits(config) * dt)

