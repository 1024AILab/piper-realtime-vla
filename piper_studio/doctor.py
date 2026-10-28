"""Explicit device check. Never enables, disables, resets, or moves a robot."""
import argparse
import json
import time

from .config import Config
from .devices import Camera, PiperHardware, discover_cameras


def main():
    parser = argparse.ArgumentParser(description="枚举设备，或只读检查 Piper 与相机")
    parser.add_argument("--config", help="指定时打开配置中的设备进行只读诊断")
    parser.add_argument("--seconds", type=float, default=3)
    args = parser.parse_args()
    print(json.dumps(discover_cameras(), ensure_ascii=False, indent=2))
    if not args.config:
        return
    config = Config.load(args.config)
    if config.mode == "sim":
        parser.error("诊断真机时请使用硬件配置，模拟模式无需设备检查")
    robot = PiperHardware(config)
    cameras = {}
    try:
        robot.connect()
        for name in ("image", "wrist_image"):
            camera = Camera(getattr(config, name), wrist=name == "wrist_image")
            cameras[name] = camera
            camera.connect()
        time.sleep(max(1, min(args.seconds, 10)))
        sample = robot.read()
        report = {"can_backend": config.can_backend,
                  "observed_ids": [hex(i) for i in sorted(robot.follower_observer.frames.frames)],
                  "leader_action_raw": sample.action.tolist() if sample.action is not None else None,
                  "follower_state_raw": sample.follower.tolist() if sample.follower is not None else None,
                  "action_age_s": sample.action_age, "feedback_age_s": sample.feedback_age,
                  "enabled": sample.enabled, "error": sample.error, "camera": {}}
        for name, camera in cameras.items():
            image, age, _ = camera.read()
            report["camera"][name] = {"shape": list(image.shape) if image is not None else None,
                                      "age_s": age, "error": camera.error}
        print(json.dumps(report, ensure_ascii=False, indent=2))
        valid = sample.action is not None and sample.follower is not None and not sample.error
        valid &= sample.action_age <= config.stale_after_s and sample.feedback_age <= config.stale_after_s
        valid &= all(c.read()[0] is not None and c.read()[1] <= config.camera_stale_after_s and not c.error for c in cameras.values())
        if not valid:
            raise SystemExit("数据未就绪；根据 observed_ids 和 error 检查接线、固件与相机配置")
    finally:
        robot.close()
        for camera in cameras.values():
            camera.close()


if __name__ == "__main__":
    main()
