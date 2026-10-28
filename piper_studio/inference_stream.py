"""Time-aligned action chunks for PiPER's seven absolute scalar coordinates.

Adapted from the scheduling / continuity / PCHIP approach in the user-supplied
robotinfer-vla-runtime-standalone. No model loading, robot I/O or delta conversion.
The model remains at 10 Hz; only the command trajectory is interpolated to 50 Hz.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator

from .motion import speed_limits, validate_targets

# Result freshness and future execution coverage are different clocks. Charging
# network time against a 2 s capture-time lease stopped two valid 1.2 s requests
# between chunks. Only a fresh result gets a new, finite execution buffer.
MAX_INFERENCE_AGE_S = 1.5
PLAN_BUFFER_S = 2.0
REPLAN_RESERVE_S = .15
MAX_CONTROL_GAP_S = 0.1
INFERENCE_TIMEOUT_MESSAGE = '新预测等待超过 1.5 秒，已停止并保持当前位置；请检查推理往返耗时'


@dataclass(frozen=True)
class StreamPlan:
    times: np.ndarray
    actions: np.ndarray
    old_weights: np.ndarray
    observed_at: float
    native_origin: float
    published_at: float
    expires_at: float
    number: int
    previous_number: int | None
    model_hz: float = 10.0

    def at(self, now):
        if now > self.expires_at:
            raise RuntimeError('连续轨迹缓存已耗尽，未收到可接续的新预测；已停止并保持当前位置')
        target = np.array([np.interp(now, self.times, self.actions[:, i]) for i in range(7)])
        old_weight = float(np.interp(now, self.times, self.old_weights))
        model_index = max(0.0, (now - self.native_origin) * self.model_hz - 1)
        return target, model_index, old_weight

    def summary(self):
        return {'control_points': len(self.actions), 'published_at': self.published_at,
                'observed_at': self.observed_at, 'expires_at': self.expires_at,
                'observation_age_at_publish_s': self.published_at - self.observed_at,
                'execution_buffer_s': self.expires_at - self.published_at,
                'previous_plan_number': self.previous_number,
                'skipped_model_steps': max(0, int((self.published_at - self.native_origin) * self.model_hz)),
                'interpolation': 'PCHIP', 'overlap': '0.1 s frozen + 0.3 s quintic blend'}


def make_stream_plan(actions, state, current_command, *, observed_at, published_at,
                     number, config, previous=None):
    """Build a detached dense plan; keep no recursive chain of old plans.

    The first plan starts at publication because the arm was holding still.
    Subsequent plans use observation time, skipping samples already in the past.
    Each result must arrive within 1.5 s of capture. Its future buffer lasts at
    most 2 s from publication, capped by the actual predicted horizon. Thus its
    observation cannot influence motion beyond 3.5 s after capture. Neither a
    pending request nor an expired plan can renew this buffer.
    """
    actions = validate_targets(actions, config)
    state = validate_targets([state], config)[0]
    current = validate_targets([current_command], config)[0]
    if not np.isfinite([observed_at, published_at]).all() or published_at < observed_at or published_at - observed_at > MAX_INFERENCE_AGE_S:
        raise RuntimeError('模型结果对应的观测已过期，未发布轨迹')
    if previous is not None and published_at >= previous.expires_at:
        raise RuntimeError('旧轨迹已过期，拒绝自动恢复运动')
    origin = published_at if previous is None else observed_at
    expires = min(published_at + PLAN_BUFFER_S, origin + len(actions) / config.fps)
    if expires <= published_at:
        raise RuntimeError('预测轨迹已没有未来动作，未发布轨迹')
    period = 1 / config.control_hz
    offsets = np.arange(0, expires - published_at, period)
    offsets = np.unique(np.append(offsets, expires - published_at))
    blend_end = min(.4, previous.expires_at - published_at, expires - published_at) if previous is not None else 0.0
    freeze_end = min(.1, blend_end / 4)
    if previous is not None:
        offsets = np.unique(np.append(offsets, [freeze_end, blend_end]))
    times = offsets + published_at
    source_times = np.arange(len(actions) + 1) / config.fps
    samples = np.vstack([current if previous is None else state, actions])
    # Clip only endpoint subtraction roundoff; times never exceed the horizon.
    query = np.clip(times - origin, 0, source_times[-1])
    values = PchipInterpolator(source_times, samples, axis=0, extrapolate=False)(query)
    weights = np.zeros(len(times))
    if previous is not None:
        u = np.clip((offsets - freeze_end) / max(blend_end - freeze_end, 1e-9), 0, 1)
        # Quintic fade has zero first/second derivatives at both ends, so it
        # does not add a velocity discontinuity at the overlap boundaries.
        weights = 1 - (10 * u**3 - 15 * u**4 + 6 * u**5)
        old_values = np.column_stack([np.interp(times, previous.times, previous.actions[:, i]) for i in range(7)])
        values = weights[:, None] * old_values + (1 - weights[:, None]) * values
    values = validate_targets(values, config)
    for a in (times, values, weights):
        a.setflags(write=False)
    return StreamPlan(times, values, weights, observed_at, origin, published_at,
                      expires, number, previous.number if previous is not None else None, config.fps)


def next_observation_at(plan, requested_interval_s):
    """Leave time for a worst-case valid reply, even after a very fast reply.

    This is a latest-start deadline, not a queued observation. The planner reads
    fresh state/images when it actually starts its next single request.
    """
    return min(plan.observed_at + requested_interval_s,
               plan.expires_at - MAX_INFERENCE_AGE_S - REPLAN_RESERVE_S)


def governed_step(current, target, velocity, dt, config):
    """Velocity and acceleration bounded commands; hard position limits win.

    A 0.2 s velocity ramp gives 25 deg/s² at 5 deg/s and 250 mm/s² at 50 mm/s.
    This limits the commanded trajectory, not measured dynamics.
    """
    if not np.isfinite(dt) or dt <= 0 or dt > MAX_CONTROL_GAP_S:
        raise RuntimeError('控制周期超过 100 ms，已停止；不突发补发积压动作')
    target = validate_targets([target], config)[0]
    current = np.asarray(current, dtype=np.float64)
    error = target - current
    vmax = speed_limits(config)
    amax = vmax / .2
    desired = np.sign(error) * np.minimum.reduce([np.abs(error) / dt, vmax, np.sqrt(2 * amax * np.abs(error))])
    velocity = np.asarray(velocity) + np.clip(desired - velocity, -amax * dt, amax * dt)
    velocity = np.clip(velocity, -vmax, vmax)
    proposed = current + velocity * dt
    low = np.array([*np.asarray(config.joint_min_deg) * 1000, 0])
    high = np.array([*np.asarray(config.joint_max_deg) * 1000, config.gripper_max_um])
    result = np.clip(proposed, low, high)
    # At a configured hard stop, retain the velocity actually allowed by the
    # bound rather than integrating an unachievable velocity into the next tick.
    velocity = np.where(result != proposed, (result - current) / dt, velocity)
    return result, velocity
