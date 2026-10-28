import json
from pathlib import Path

import numpy as np
import pytest

from piper_studio.config import Config
from piper_studio.inference_stream import (MAX_INFERENCE_AGE_S, PLAN_BUFFER_S,
    REPLAN_RESERVE_S, governed_step, make_stream_plan, next_observation_at)

Q = np.array([0, 30000, -45000, 0, 25000, 0, 35000], dtype=np.float64)
CFG = Config(max_joint_speed_deg_s=5, max_gripper_speed_mm_s=15)


def plan(actions, observed=10., published=10.2, previous=None, number=1):
    return make_stream_plan(actions, Q, Q, observed_at=observed,
        published_at=published, previous=previous, number=number, config=CFG)


def test_pchip_interpolates_all_seven_absolute_scalars_without_overshoot():
    actions = Q[None] + np.arange(1, 51)[:, None] * np.array([100, 50, -30, 20, 10, -40, 150])
    p = plan(actions)
    np.testing.assert_allclose(p.at(10.2)[0], Q)
    np.testing.assert_allclose(p.at(10.3)[0], actions[0])
    np.testing.assert_allclose(p.at(10.4)[0], actions[1])
    assert len(p.actions) >= 90 and not p.actions.flags.writeable
    assert np.all(p.actions >= np.minimum(Q, actions.min(axis=0)) - 1e-7)
    assert np.all(p.actions <= np.maximum(Q, actions.max(axis=0)) + 1e-7)


def test_new_plan_keeps_old_trajectory_then_blends_and_skips_expired_prefix():
    old = plan(Q[None] + np.arange(1, 51)[:, None] * 50)
    new_actions = Q[None] + np.arange(1, 51)[:, None] * 200
    new = plan(new_actions, observed=10.6, published=10.85, previous=old, number=2)
    np.testing.assert_allclose(new.at(10.85)[0], old.at(10.85)[0])
    np.testing.assert_allclose(new.at(10.93)[0], old.at(10.93)[0])
    assert new.at(11.05)[2] < 1 and new.at(11.05)[2] > 0
    target, _, weight = new.at(11.3)
    assert weight == 0
    np.testing.assert_allclose(target, Q + 7 * 200, atol=1e-7)
    assert new.summary()['skipped_model_steps'] == 2
    assert new.previous_number == 1


def test_old_influence_cannot_outlive_its_fixed_execution_buffer():
    old = plan(np.repeat((Q + 500)[None], 50, axis=0))
    new = plan(np.repeat((Q - 500)[None], 50, axis=0), observed=11.9,
               published=12.05, previous=old, number=2)
    assert old.expires_at == 12.2
    assert new.at(old.expires_at)[2] == 0
    assert new.at(12.3)[2] == 0
    np.testing.assert_allclose(new.at(12.3)[0], Q - 500)
    with pytest.raises(RuntimeError, match='缓存已耗尽'):
        new.at(new.expires_at + .01)
    with pytest.raises(RuntimeError, match='拒绝自动恢复'):
        plan(np.repeat(Q[None], 50, axis=0), observed=12.1, published=12.2, previous=old)


@pytest.mark.parametrize('delays,interval', [
    ([1.213] * 10, .5),  # Exact failure regime: consecutive replies exceed 2 s.
    ([1.49] * 10, .5),
    ([.02, 1.49, .06, 1.4] * 3, 1.0),  # Fast reply must schedule slow successor early.
])
def test_valid_back_to_back_delayed_results_keep_future_coverage(delays, interval):
    previous = None
    observed = 10.
    actions = np.repeat((Q + 500)[None], 50, axis=0)
    for number, delay in enumerate(delays, 1):
        published = observed + delay
        if previous is not None:
            # Control remains defined throughout the actual network wait.
            previous.at(published)
            assert previous.expires_at - published >= REPLAN_RESERVE_S - .021
        new = plan(actions, observed=observed, published=published,
                   previous=previous, number=number)
        assert new.expires_at == pytest.approx(published + PLAN_BUFFER_S)
        assert new.expires_at <= observed + MAX_INFERENCE_AGE_S + PLAN_BUFFER_S
        if previous is not None:
            np.testing.assert_allclose(new.at(published)[0], previous.at(published)[0])
        observed = max(published, next_observation_at(new, interval)) + .02
        previous = new


def test_buffer_never_extrapolates_beyond_real_model_horizon():
    old = plan(np.repeat(Q[None], 50, axis=0))
    short = plan(np.repeat(Q[None], 5, axis=0), observed=10.5,
                 published=10.7, previous=old, number=2)
    assert short.expires_at == 11.0
    with pytest.raises(RuntimeError, match='缓存已耗尽'):
        short.at(11.01)


def test_governor_bounds_velocity_and_acceleration_when_target_reverses():
    q, velocity = Q.copy(), np.zeros(7)
    rows, velocities = [q.copy()], []
    for i in range(200):
        target = Q + (10000 if i < 80 else -10000)
        q, velocity = governed_step(q, target, velocity, .02, CFG)
        rows.append(q.copy()); velocities.append(velocity.copy())
    velocities = np.array(velocities)
    assert np.max(np.abs(velocities[:, :6])) <= 5000
    assert np.max(np.abs(velocities[:, 6])) <= 15000
    accelerations = np.diff(velocities, axis=0) / .02
    assert np.max(np.abs(accelerations[:, :6])) <= 25000 + 1e-6
    assert np.max(np.abs(accelerations[:, 6])) <= 75000 + 1e-6
    assert np.max(np.abs(np.diff(rows, axis=0)[:, :6])) <= 100 + 1e-6


def test_stale_result_nonfinite_target_and_control_stall_are_rejected():
    with pytest.raises(RuntimeError, match='过期'):
        plan(np.repeat(Q[None], 50, axis=0), observed=10, published=11.501)
    with pytest.raises(RuntimeError, match='控制周期'):
        governed_step(Q, Q, np.zeros(7), .101, CFG)
    bad = Q.copy(); bad[2] = np.nan
    with pytest.raises(ValueError):
        governed_step(Q, bad, np.zeros(7), .02, CFG)


def test_recorded_release_pulses_reach_past_object_width_at_inference_speed():
    fixture = json.loads((Path(__file__).parent / 'fixtures/gripper-release-window.json').read_text())
    # Model targets come from the real failed release. At that time actual
    # opening was ~23 mm: commands below that width kept squeezing the duck.
    peaks = {}
    for speed in (15, 50):
        cfg = Config(max_joint_speed_deg_s=5, max_gripper_speed_mm_s=speed)
        q, v = Q.copy(), np.zeros(7)
        q[6] = 0
        commands = []
        for dt, width in fixture['samples']:
            target = Q.copy(); target[6] = width
            old_q = q.copy()
            q, v = governed_step(q, target, v, dt, cfg)
            q = np.rint(q)
            assert abs(q[6] - old_q[6]) <= speed * 1000 * dt + 1
            np.testing.assert_array_equal(q[:6], Q[:6])
            commands.append(q[6])
        assert min(commands) >= 0 and max(commands) <= 70000
        peaks[speed] = max(commands)
    assert peaks[15] < 20000  # Reproduce suppressed opening before the fix.
    assert peaks[50] > 40000  # Respond to the same pulse; not a hardware success claim.
