import json
import threading
import time

import numpy as np
import pytest

from piper_studio.config import Config
from piper_studio.devices import ArmSample
from piper_studio.inference import InferenceEngine, PROMPT, bounded_prediction
from piper_studio.inference_stream import make_stream_plan
from piper_studio.policy_client import DemoPolicy, encode_array, unpack
import msgpack


class CameraStub:
    error = ''
    def read(self):
        return np.zeros((480, 640, 3), dtype=np.uint8), 0, time.monotonic()
    def close(self):
        pass


class RobotStub:
    def __init__(self):
        self.q = np.array([0, 30000, -45000, 0, 25000, 0, 35000], dtype=np.float32)
        self.sent = []
    def read(self):
        return ArmSample(None, self.q.copy(), float('inf'), 0, True)
    def external_control_recent(self):
        return False
    def send(self, q):
        self.q = np.rint(q).astype(np.float32)
        self.sent.append(self.q.copy())
        return self.q.copy()
    def close(self):
        pass


@pytest.fixture
def engine(tmp_path):
    e = InferenceEngine(Config(max_joint_speed_deg_s=5), tmp_path, DemoPolicy())
    e.shutdown.set()
    e.worker.join(2); e.planner.join(2)
    e.robot = RobotStub()
    e.cameras = {'image': CameraStub(), 'wrist_image': CameraStub()}
    e.connected = True
    e.sample = e.robot.read()
    e.sample_at = time.monotonic()
    e.model = e.client.check()
    yield e
    e.close()


def start(e, mode='preview'):
    e.start(mode, 'test', PROMPT, 60, 3, 5, 'browser-1')


def ready_execute(e):
    e.prepared = True
    e.owner = 'browser-1'
    e.last_sent = e.sample.follower.copy()
    now = time.monotonic()
    e.trace.extend([(now - .2, e.last_sent.copy()), (now, e.last_sent.copy())])
    start(e, 'executing')


def test_feedback_without_leader_is_sufficient_and_preview_never_sends(engine):
    assert engine.sample.action is None
    start(engine)
    engine._control_tick(time.monotonic())
    engine.stop()
    assert not engine.robot.sent
    assert not engine.prepared


def test_cannot_execute_without_explicit_prepare(engine):
    with pytest.raises(RuntimeError, match='准备'):
        start(engine, 'executing')
    assert not engine.robot.sent


def install_plan(engine, now, actions=None):
    actions = np.repeat(engine.last_sent[None], 50, axis=0) if actions is None else actions
    engine.active_plan = make_stream_plan(actions, engine.last_sent, engine.last_sent,
        observed_at=now, published_at=now, number=1, config=engine.config)


def test_continuous_control_limits_speed_and_preserves_actual_sent_state(engine):
    ready_execute(engine)
    initial = engine.last_sent.copy()
    now = time.monotonic()
    install_plan(engine, now, np.repeat((initial + 3000)[None], 50, axis=0))
    for i in range(50):
        engine.sample = engine.robot.read()
        engine._control_tick(now + i * .02)
        np.testing.assert_array_equal(engine.last_sent, engine.robot.sent[-1])
    changes = np.diff(np.vstack([initial, engine.robot.sent]), axis=0)
    assert np.max(np.abs(changes[:, :6])) <= 101  # 5 deg/s * 20 ms plus rounding.
    assert engine.active_plan is not None
    assert len(engine.robot.sent) == 50  # No pause after the old three-step boundary.


def test_stop_clears_buffer_and_no_commands_after_stop(engine):
    ready_execute(engine)
    install_plan(engine, time.monotonic())
    engine.stop()
    after_stop = len(engine.robot.sent)
    assert engine.active_plan is None and not engine.prepared
    engine._control_tick(time.monotonic())
    assert len(engine.robot.sent) == after_stop


def test_stale_browser_and_observation_block_send(engine):
    ready_execute(engine)
    engine.heartbeat_at = time.monotonic() - 4
    with pytest.raises(RuntimeError, match='心跳'):
        engine._control_tick(time.monotonic())
    assert not engine.robot.sent
    engine.heartbeat_at = time.monotonic()
    install_plan(engine, time.monotonic() - 3)
    with pytest.raises(RuntimeError, match='缓存已耗尽'):
        engine._control_tick(time.monotonic())
    assert not engine.robot.sent


def test_other_browser_cannot_take_over_or_renew_lease(engine):
    start(engine)
    engine.heartbeat_at = 1
    engine.heartbeat('browser-2')
    assert engine.heartbeat_at == 1
    with pytest.raises(RuntimeError, match='另一个页面'):
        engine._claim('browser-2')


def test_stale_camera_and_external_controller_block_start(engine):
    engine.cameras['image'].error = 'offline'
    with pytest.raises(RuntimeError, match='offline'):
        start(engine)
    engine.cameras['image'].error = ''
    engine.robot.external_control_recent = lambda: True
    with pytest.raises(RuntimeError, match='外部 CAN'):
        start(engine)
    assert not engine.robot.sent


def test_msgpack_round_trip_preserves_raw_units_and_rgb():
    obs = {'state': np.arange(7, dtype=np.float32) * 1000,
           'image': np.array([[[255, 0, 4]]], dtype=np.uint8), 'prompt': PROMPT}
    decoded = unpack(msgpack.packb(obs, default=encode_array))
    np.testing.assert_array_equal(obs['state'], decoded['state'])
    np.testing.assert_array_equal(obs['image'], decoded['image'])


def test_late_response_cannot_resurrect_stopped_run(tmp_path):
    entered, release = threading.Event(), threading.Event()
    class SlowPolicy(DemoPolicy):
        def infer(self, observation, timeout=2):
            entered.set(); release.wait(2)
            return np.repeat(observation['state'][None], 50, axis=0), {}
    e = InferenceEngine(Config(), tmp_path, SlowPolicy())
    try:
        e.connect(); e.model = e.client.check()
        time.sleep(.08)
        start(e)
        assert entered.wait(1)
        e.stop()
        release.set()
        time.sleep(.1)
        assert e.prediction is None
        assert e.active_plan is None
        assert e.robot.send_count == 0
    finally:
        release.set(); e.close()


def test_replay_is_read_only(engine):
    start(engine)
    state = engine.sample.follower.tolist()
    engine._event('prediction', state=state, actions=[state] * 50, number=1, latency_ms=100)
    engine.stop()
    row, images, count, index = engine.replay(engine.run_dir.name)
    assert count == 1 and row['state'] == state
    assert not engine.robot.sent


def test_gripper_is_bounded_without_modifying_raw_prediction(engine):
    state = engine.sample.follower
    actions = np.repeat(state[None], 50, axis=0)
    actions[:, 6] = np.linspace(-1500, 100000, 50)
    checked, note = bounded_prediction(actions, state, engine.config)
    assert actions[-1, 6] == 100000 and actions[0, 6] == -1500
    assert checked[:, 6].min() == 0 and checked[:, 6].max() == 70000
    assert '限制' in note
    actions[-1, 6] = 200000
    assert bounded_prediction(actions, state, engine.config)[0][-1, 6] == 70000
    actions[-1, 0] = 200000
    with pytest.raises(ValueError, match='关节范围'):
        bounded_prediction(actions, state, engine.config)


def test_gripper_speed_is_independent_and_logged_for_the_next_run(engine):
    engine.start('preview', 'gripper-speed', PROMPT, 0, 5, 5, 'browser-1', 50)
    assert engine.config.max_gripper_speed_mm_s == 50
    assert engine.config.max_joint_speed_deg_s == 5
    assert engine.config.gripper_max_um == 70000
    meta = json.loads((engine.run_dir / 'meta.json').read_text())
    assert meta['config']['max_gripper_speed_mm_s'] == 50
    # Starting another run must not change the speed of the current run.
    with pytest.raises(RuntimeError, match='暂停当前运行'):
        engine.start('preview', 'other', PROMPT, 0, 5, 10, 'browser-1', 100)
    assert engine.config.max_gripper_speed_mm_s == 50
    assert engine.config.max_joint_speed_deg_s == 5
    assert not engine.robot.sent


@pytest.mark.parametrize('speed', [float('nan'), float('inf'), 0, 101])
def test_invalid_gripper_speed_cannot_start_or_send(engine, speed):
    with pytest.raises(ValueError, match='夹爪速度'):
        engine.start('preview', 'bad-speed', PROMPT, 0, 5, 5, 'browser-1', speed)
    assert engine.phase == 'idle' and not engine.robot.sent


def test_capture_and_inference_cannot_claim_same_can(monkeypatch):
    from piper_studio import devices
    class Observer:
        def __init__(self, *a):
            pass
        def close(self):
            pass
    monkeypatch.setattr(devices, '_CanObserver', Observer)
    cfg = Config(mode='hardware_bus', can_backend='gs_usb', follower_port='99881')
    a, b = devices.PiperHardware(cfg), devices.PiperHardware(cfg)
    try:
        a.connect()
        with pytest.raises(RuntimeError, match='另一个 Piper Studio'):
            b.connect()
        a.close()
        b.connect()
    finally:
        a.close(); b.close()


def test_manual_run_does_not_stop_at_sixty_seconds(engine):
    engine.start('preview', 'manual', PROMPT, 0, 5, 5, 'browser-1')
    now = engine.run_started + 65
    engine.heartbeat_at = now
    engine._control_tick(now)
    assert engine.phase == 'preview'
    assert engine.snapshot()['remaining'] is None
    assert not engine.robot.sent
    engine.stop()
    assert engine.phase == 'idle'


def test_timed_stop_explains_not_task_completion(engine):
    start(engine)
    engine.heartbeat_at = engine.deadline
    engine._control_tick(engine.deadline)
    assert engine.phase == 'idle'
    assert '不代表任务完成' in engine.last_stop_reason


def stationary(engine):
    now = time.monotonic()
    engine.sample = engine.robot.read()
    engine.sample_at = now
    engine.trace.clear()
    engine.trace.extend([(now - .2, engine.sample.follower.copy()), (now, engine.sample.follower.copy())])


def test_remember_home_is_passive_and_persists(engine):
    stationary(engine)
    engine.remember_home()
    expected = engine.sample.follower.copy()
    assert not engine.robot.sent
    engine.home = None
    engine._restore_home()
    np.testing.assert_array_equal(engine.home, expected)
    assert json.loads((engine.run_root / 'home.json').read_text())['identity'] == engine._home_identity()


def test_home_recovers_first_state_not_final_command(engine):
    ready_execute(engine)
    original = engine.last_sent.copy()
    engine._event('prediction', state=original.tolist())
    engine._event('command', sent=(original + 100).tolist())
    engine.stop()
    (engine.run_root / 'home.json').unlink()
    engine.home = None
    engine._restore_home()
    np.testing.assert_array_equal(engine.home, original)
    # A history from another CAN/camera configuration cannot become home.
    (engine.run_root / 'home.json').unlink()
    engine.home = None
    engine.config.follower_port = 'another-port'
    engine._restore_home()
    assert engine.home is None


def test_home_requires_confirmation_and_enabled_arm(engine):
    stationary(engine)
    engine.remember_home()
    with pytest.raises(ValueError, match='勾选'):
        engine.go_home(False, 'browser-1')
    engine.sample.enabled = False
    with pytest.raises(RuntimeError, match='使能'):
        engine.go_home(True, 'browser-1')
    assert not engine.robot.sent


def test_home_tracks_feedback_limits_speed_and_can_stop(engine):
    stationary(engine)
    engine.remember_home()
    engine.robot.q[0] += 5000
    stationary(engine)
    engine.cameras = {}  # Returning home does not need the model or cameras.
    engine.go_home(True, 'browser-1')
    origin = engine.last_sent.copy()
    began = engine.home_started
    previous = origin
    for i in range(1, 120):
        now = began + i * .02
        engine.sample = engine.robot.read()
        engine.sample_at = time.monotonic()
        engine.heartbeat_at = now
        engine._control_tick(now)
        if engine.robot.sent:
            sent = engine.robot.sent[-1]
            assert np.max(np.abs(sent[:6] - previous[:6])) <= 101
            previous = sent.copy()
        if engine.phase == 'idle':
            break
    assert engine.phase == 'idle'
    assert engine.prepared
    np.testing.assert_allclose(engine.robot.q, engine.home, atol=3)
    assert '已回到' in engine.last_stop_reason
    stationary(engine)
    engine.go_home(True, 'browser-1')
    engine.stop()
    count = len(engine.robot.sent)
    engine._control_tick(time.monotonic())
    assert len(engine.robot.sent) == count and engine.home_motion is None


def test_home_does_not_report_success_without_actual_arrival(engine):
    stationary(engine)
    engine.remember_home()
    engine.robot.q[0] += 5000
    stationary(engine)
    engine.go_home(True, 'browser-1')
    # Simulate a mechanically stalled arm whose command is already at home.
    engine.robot.send = lambda q: np.asarray(q).copy()
    engine.last_sent = engine.home.copy()
    now = engine.home_started + engine.home_motion.duration + 1
    engine.heartbeat_at = now
    engine._control_tick(now)
    assert engine.phase == 'returning'
    now += 5
    engine.heartbeat_at = now
    with pytest.raises(RuntimeError, match='实际反馈到位超时'):
        engine._control_tick(now)


def test_home_accepts_declared_encoder_tolerance_and_reports_error(engine):
    stationary(engine)
    engine.remember_home()
    engine.robot.q[0] += 2000
    stationary(engine)
    engine.go_home(True, 'browser-1')
    engine.robot.send = lambda q: np.asarray(q).copy()
    engine.last_sent = engine.home.copy()
    now = engine.home_started + engine.home_motion.duration + .1
    engine.heartbeat_at = now
    engine._control_tick(now)
    assert engine.phase == 'idle'
    assert '2.00°' in engine.last_stop_reason
    assert '允许 2.5°' in engine.last_stop_reason


def test_duplicate_ticks_and_long_control_gap_cannot_burst_commands(engine):
    ready_execute(engine)
    now = time.monotonic()
    install_plan(engine, now)
    engine._control_tick(now)
    engine._control_tick(now + .002)
    engine._control_tick(now + .008)
    assert len(engine.robot.sent) == 1
    with pytest.raises(RuntimeError, match='控制周期'):
        engine._control_tick(now + .2)
    assert len(engine.robot.sent) == 1


def test_inference_in_flight_does_not_pause_control_and_uses_latest_sent_state(tmp_path):
    entered, release = threading.Event(), threading.Event()
    class DelayedPolicy(DemoPolicy):
        calls = 0
        def infer(self, observation, timeout=2):
            self.calls += 1
            if self.calls == 2:
                entered.set()
                release.wait(1)
            else:
                time.sleep(.16)
            actions = np.repeat(observation['state'][None], 50, axis=0)
            actions[:, 0] += np.arange(1, 51) * 150
            return actions, {'infer_ms': 160}
    e = InferenceEngine(Config(max_joint_speed_deg_s=5), tmp_path, DelayedPolicy())
    try:
        e.connect(); e.model = e.client.check()
        time.sleep(.2)
        e.prepare(True, 'browser-1')
        limit = time.monotonic() + 1
        while not e.prepared and time.monotonic() < limit:
            time.sleep(.02)
        e.start('executing', 'async-test', PROMPT, 0, 5, 5, 'browser-1')
        assert entered.wait(1.5)
        before = e.robot.send_count
        time.sleep(.3)
        e.heartbeat('browser-1')
        assert e.inflight and e.phase == 'executing'
        assert e.robot.send_count - before >= 10
        release.set()
        limit = time.monotonic() + 1
        while e.plan_count < 2 and time.monotonic() < limit:
            time.sleep(.02)
        assert e.plan_count >= 2
        e.stop()
        rows = [json.loads(s) for s in (e.run_dir / 'events.jsonl').read_text().splitlines()]
        second = [r for r in rows if r['event'] == 'prediction'][1]
        previous_commands = [r for r in rows if r['event'] == 'command' and r['monotonic_time'] <= second['observed_at']]
        np.testing.assert_array_equal(second['state'], previous_commands[-1]['sent'])
        assert second['stream']['previous_plan_number'] == 1
    finally:
        release.set(); e.close()


def test_consecutive_slow_predictions_continue_but_hung_successor_stops_independently(tmp_path):
    """Reproduce the user's 1.21 s first reply and a similarly slow second one."""
    hung, release = threading.Event(), threading.Event()
    class SlowThenHungPolicy(DemoPolicy):
        calls = 0
        def infer(self, observation, timeout=2):
            self.calls += 1
            if self.calls == 4:
                hung.set()
                release.wait(4)  # Deliberately ignore the client timeout.
            else:
                time.sleep([1.213, 1.213, 1.38][self.calls - 1])
            actions = np.repeat(observation['state'][None], 50, axis=0)
            actions[:, 0] += np.arange(1, 51) * 50
            return actions, {'infer_ms': 130}
    e = InferenceEngine(Config(max_joint_speed_deg_s=5), tmp_path, SlowThenHungPolicy())
    try:
        e.connect(); e.model = e.client.check()
        time.sleep(.2)
        e.prepare(True, 'browser-1')
        limit = time.monotonic() + 1
        while not e.prepared and time.monotonic() < limit:
            time.sleep(.02)
        e.start('executing', 'slow-network-regression', PROMPT, 0, 5, 5, 'browser-1')
        limit = time.monotonic() + 5
        while not hung.is_set() and e.phase == 'executing' and time.monotonic() < limit:
            e.heartbeat('browser-1'); time.sleep(.02)
        assert hung.is_set() and e.plan_count == 3 and e.phase == 'executing', e.error
        assert e.sent_count >= 115  # First/second/third chunks have kept moving.
        deadline = e.inference_deadline
        limit = deadline + .3
        while e.phase == 'executing' and time.monotonic() < limit:
            e.heartbeat('browser-1'); time.sleep(.02)
        assert e.phase == 'fault' and '新预测等待超过 1.5 秒' in e.error
        assert e.inflight and not release.is_set()  # Control stopped without the planner returning.
        assert e.active_plan is None and not e.prepared
        assert deadline <= e.run_ended < deadline + .15
        count = e.robot.send_count
        release.set()
        limit = time.monotonic() + .5
        while e.inflight and time.monotonic() < limit:
            time.sleep(.01)
        assert not e.inflight and e.plan_count == 3 and e.robot.send_count == count
        rows = [json.loads(s) for s in (e.run_dir / 'events.jsonl').read_text().splitlines()]
        commands = [r for r in rows if r['event'] == 'command']
        assert max(np.diff([r['monotonic_time'] for r in commands])) < .1
        assert len([r for r in rows if r['event'] == 'inference_request']) == 4
    finally:
        release.set(); e.close()
