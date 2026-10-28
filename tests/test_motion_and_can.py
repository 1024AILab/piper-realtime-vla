import struct
from types import SimpleNamespace

import numpy as np
import pytest

from piper_studio.config import Config
from piper_studio.devices import CanFrames, PiperHardware, command_frames
from piper_studio.motion import Trajectory, limited_step, speed_limits, validate_targets


def test_can_needs_all_four_frames_and_tracks_oldest_age():
    c = CanFrames()
    for i in range(3):
        c.feed(0x155 + i, struct.pack(">ii", i, -i), now=10)
    assert c.read(True, now=10.2)[0] is None
    c.feed(0x159, struct.pack(">iHBB", -70, 1000, 1, 0), now=10.1)
    q, age = c.read(True, now=10.3)
    np.testing.assert_array_equal(q, [0, 0, 1, -1, 2, -2, -70])
    assert age == pytest.approx(0.3)
    c.feed(0x155, struct.pack(">ii", 10, 20), now=20)
    assert c.read(True, now=20)[1] == 10  # Updating J1/J2 cannot hide stale J3–J6.


def test_partial_targets_keep_missing_channels_missing_and_expose_each_frame_age():
    c = CanFrames()
    c.feed(0x155, struct.pack(">ii", 1000, 2000), now=10)
    q, ages = c.read_parts(True, now=10.1)
    np.testing.assert_array_equal(q[:2], [1000, 2000])
    assert np.isnan(q[2:]).all()
    assert ages["0x155"] == pytest.approx(0.1)
    assert ages["0x156"] is None and ages["0x159"] is None
    assert c.read(True, now=10.1)[0] is None


def test_can_local_echo_does_not_block_replay_external_command_does():
    c = CanFrames()
    c.feed(0x155, bytes(8), is_rx=False, now=100)
    assert c.external_control_at == 0
    c.feed(0x151, bytes(8), is_rx=True, now=101)
    assert c.external_control_at == 101


def feed_status(c, at=10, flags=0x40):
    c.feed(0x2A1, bytes(8), now=at)
    for i in range(6):
        c.feed(0x261 + i, bytes([0, 0, 0, 0, 0, flags, 0, 0]), now=at)


def test_status_enable_fault_and_staleness_fail_closed():
    c = CanFrames()
    assert c.status(0.5, now=10)[1]
    feed_status(c)
    assert c.status(0.5, now=10.1) == (True, "")
    assert c.status(0.5, now=11)[0] is False
    feed_status(c, flags=0xC0)
    assert "故障" in c.status(0.5, now=10.1)[1]
    feed_status(c, flags=0)
    assert c.status(0.5, now=10.1) == (False, "")


def test_wire_commands_match_official_sdk_encoder():
    import can
    from piper_sdk.protocol.protocol_v2.piper_protocol_v2 import C_PiperParserV2
    from piper_sdk.piper_msgs.msg_v2 import PiperMessage, ArmMsgType
    parser = C_PiperParserV2()
    msg = PiperMessage()
    msg.arm_motion_ctrl_2.ctrl_mode = 1
    msg.arm_motion_ctrl_2.move_mode = 1
    msg.arm_motion_ctrl_2.move_spd_rate_ctrl = 10
    msg.arm_motion_ctrl_2.mit_mode = 0xAD
    q = [1000, 30000, -50000, -4000, 20000, 6000, 35000]
    for i, value in enumerate(q[:6], 1):
        setattr(msg.arm_joint_ctrl, f"joint_{i}", value)
    msg.arm_gripper_ctrl.grippers_angle = q[6]
    msg.arm_gripper_ctrl.grippers_effort = 1000
    msg.arm_gripper_ctrl.status_code = 1
    msg.arm_gripper_ctrl.set_zero = 0
    names = ["PiperMsgMotionCtrl_2", "PiperMsgJointCtrl_12", "PiperMsgJointCtrl_34", "PiperMsgJointCtrl_56", "PiperMsgGripperCtrl"]
    expected = []
    for name in names:
        msg.type_ = getattr(ArmMsgType, name)
        frame = can.Message()
        assert parser.EncodeMessage(msg, frame)
        expected.append((frame.arbitration_id, bytes(frame.data)))
    assert command_frames(q, Config()) == expected
    assert all(identifier != 0x150 for identifier, _ in expected)  # Never reset/disable torque.


def test_hardware_connect_is_passive_and_close_never_disables(monkeypatch):
    import piper_studio.devices as module
    observers = []
    class Observer:
        def __init__(self, *args):
            self.frames, self.error, self.sent = CanFrames(), "", []
            observers.append(self)
        def send(self, *args):
            self.sent.append(args)
        def close(self):
            pass
    monkeypatch.setattr(module, "_CanObserver", Observer)
    # Mock observer must not contend with the user's live adapter 0 lease.
    c = Config(can_backend="gs_usb", follower_port="99882")
    hardware = PiperHardware(c)
    hardware.connect()
    with pytest.raises(RuntimeError, match="静默观察"):
        hardware.prepare()
    hardware.close()
    assert len(observers) == 1
    assert observers[0].sent == []


def test_hardware_send_returns_exact_transmitted_integer_target():
    written = []
    hardware = PiperHardware(Config())
    hardware.follower_observer = SimpleNamespace(send=lambda *args: written.append(args))
    q = [1234.45, 30000.76, -40000.33, 0, 20000, 0, -70]
    result = hardware.send(q)
    np.testing.assert_array_equal(result, [1234, 30001, -40000, 0, 20000, 0, 0])
    assert len(written) == 5


def test_trajectory_respects_speed_and_approaches_first_pose_smoothly():
    config = Config()
    q = np.asarray([0, 30000, -50000, 0, 20000, 0, 30000])
    actions = np.stack([q + [50000, 0, 0, 0, 0, 0, 0], q + [60000, 0, 0, 0, 0, 0, 0]])
    plan = Trajectory(q, actions, config, 1.0)
    assert plan.speed < 1
    times = np.arange(0, plan.duration, 0.002)
    path = np.stack([plan.at(t)[0] for t in times])
    assert np.max(np.abs(np.diff(path, axis=0)) / 0.002 / speed_limits(config)) < 1.0001
    np.testing.assert_array_equal(plan.at(0)[0], q)
    np.testing.assert_array_equal(plan.at(plan.duration + 1)[0], actions[-1])
    assert plan.at(plan.duration + 1)[2]


@pytest.mark.parametrize("bad", [[0] * 6, [[float("nan")] * 7], [[999000] * 7]])
def test_invalid_actions_rejected(bad):
    with pytest.raises(ValueError):
        validate_targets(bad, Config())


def test_small_negative_gripper_noise_only_clamped_for_execution():
    q = np.asarray([[0, 30000, -50000, 0, 20000, 0, -70]])
    assert validate_targets(q, Config())[0, 6] == 0
    assert q[0, 6] == -70
