"""One operator, one CAN owner. Network I/O never holds the motion lock.

Read-only connect/preview; explicit enable/start; fresh feedback/cameras, browser
lease, bounded action horizon, cancellation generation, no motion auto-retry.
All state/action vectors are native PiPER 0.001 degree / micrometer units.
"""
from __future__ import annotations

import collections
import json
import re
import threading
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from .config import Config
from .devices import Camera, PiperHardware, SimRobot
from .motion import Trajectory, limited_step, validate_targets
from .inference_stream import (INFERENCE_TIMEOUT_MESSAGE, MAX_CONTROL_GAP_S,
    MAX_INFERENCE_AGE_S, PLAN_BUFFER_S, governed_step, make_stream_plan, next_observation_at)
from .policy_client import PolicyClient, DemoPolicy

PROMPT = 'Pick up the yellow duck and place it in the pen holder.'
# Explicit return tolerance; does not alter encoder calibration or joint limits.
HOME_TOLERANCE = np.array([2500] * 6 + [1000], dtype=np.float64)


def bounded_prediction(actions, state, config):
    """Keep model outputs intact; bound gripper width before target validation."""
    a = np.asarray(actions, dtype=np.float64).copy()
    if a.shape != (50, 7) or not np.isfinite(a).all():
        raise ValueError('模型输出不是有效的 50 × 7 动作')
    low, high = float(a[:, 6].min()), float(a[:, 6].max())
    a[:, 6] = np.clip(a[:, 6], 0, config.gripper_max_um)
    checked = validate_targets(a, config)
    if np.max(np.abs(checked[0, :6] - state[:6])) > 30000:
        raise ValueError('首个预测距当前 state 超过 30°，请核对任务起始姿态')
    note = (f'夹爪预测 {low / 1000:.1f}–{high / 1000:.1f} mm；执行开度限制为 0–{config.gripper_max_um / 1000:.0f} mm'
            if low < 0 or high > config.gripper_max_um else '')
    return checked, note


class InferenceEngine:
    def __init__(self, config: Config, run_root='runtime/inference-runs', client=None):
        config.validate()
        if config.mode == 'software':
            raise ValueError('推理只使用一个从臂 CAN 接口')
        if config.mode != 'sim' and isinstance(client, DemoPolicy):
            raise ValueError('模拟预测不能连接真机')
        self.config = config
        self.run_root = Path(run_root).resolve()
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.client = client or (DemoPolicy() if config.mode == 'sim' else PolicyClient())
        self.lock = threading.RLock()
        self.operation_lock = threading.Lock()
        self.shutdown = threading.Event()
        self.robot = None
        self.cameras = {}
        self.connected = False
        self.sample = None
        self.sample_at = 0.0
        self.prepared = False
        self.phase = 'idle'
        self.generation = 0
        self.heartbeat_at = 0.0
        self.owner = None
        self.last_sent = None
        self.prediction = None
        self.prediction_at = 0.0
        self.prediction_note = ''
        self.active_plan = None
        self.chunk_steps = 5
        self.command_velocity = np.zeros(7)
        self.last_control_send_at = None
        self.missed_control_ticks = 0
        self.next_infer = 0.0
        self.deadline = 0.0
        self.prepare_deadline = 0.0
        self.next_enable = 0.0
        self.inflight = False
        self.inference_deadline = None
        self.latency_ms = None
        self.server_ms = None
        self.plan_count = 0
        self.sent_count = 0
        self.run_started = 0.0
        self.run_ended = 0.0
        self.last_stop_reason = ''
        self.home = None
        self.home_source = ''
        self.home_motion = None
        self.home_started = 0.0
        self.home_last_tick = 0.0
        self.error = ''
        self.model = {}
        self.prompt = PROMPT
        self.trace = collections.deque(maxlen=200)
        self.logs = collections.deque(maxlen=100)
        self.run_dir = None
        self.journal = None
        self._restore_home()
        self.worker = threading.Thread(target=self._control_loop, name='inference-control', daemon=True)
        self.planner = threading.Thread(target=self._planner_loop, name='inference-planner', daemon=True)
        self.worker.start()
        self.planner.start()
        self.log('推理平台已启动。连接设备与预览预测均不会发送运动命令。')

    def log(self, text):
        self.logs.append(f'{time.strftime("%H:%M:%S")}  {text}')

    def _event(self, kind, **data):
        if self.journal:
            self.journal.write(json.dumps({'event': kind, 'time': time.time(), **data}, ensure_ascii=False) + '\n')
            self.journal.flush()

    def _home_identity(self):
        return [self.config.mode, self.config.can_backend, self.config.follower_port,
                self.config.image.device, self.config.wrist_image.device]

    def _write_home(self, raw, source):
        self.home = validate_targets([raw], self.config)[0]
        self.home_source = source
        p = self.run_root / 'home.json'
        temporary = p.with_suffix('.tmp')
        temporary.write_text(json.dumps({'raw': self.home.tolist(), 'source': source,
            'identity': self._home_identity(), 'saved_at': time.time()}, ensure_ascii=False, indent=2))
        temporary.replace(p)

    def _restore_home(self):
        p = self.run_root / 'home.json'
        if p.exists():
            try:
                data = json.loads(p.read_text())
                if data['identity'] == self._home_identity():
                    self.home = validate_targets([data['raw']], self.config)[0]
                    self.home_source = data['source']
                    return
            except (ValueError, KeyError, OSError):
                pass
        # Recover the actual command state at the beginning of the user's last
        # execution; do not substitute the arm's current post-run pose as home.
        for p in sorted(self.run_root.glob('*/meta.json'), reverse=True):
            try:
                meta = json.loads(p.read_text())
                cfg = meta['config']
                identity = [cfg['mode'], cfg['can_backend'], cfg['follower_port'],
                            cfg['image']['device'], cfg['wrist_image']['device']]
                if meta['mode'] != 'executing' or identity != self._home_identity():
                    continue
                with p.with_name('events.jsonl').open() as file:
                    for line in file:
                        event = json.loads(line)
                        if event['event'] == 'prediction':
                            self._write_home(event['state'], '恢复自上次推理的起始姿态：' + p.parent.name)
                            return
            except (ValueError, KeyError, OSError):
                continue

    def remember_home(self):
        with self.lock:
            if self.phase in {'executing', 'preview', 'preparing', 'returning'} or self.inflight:
                raise RuntimeError('请先停止当前运行，再记住起始位')
            self._health(cameras=False)
            self._stationary()
            self._write_home(self.sample.follower, '操作员记住的起始姿态')
            self.log('已记住当前起始位；没有发送运动指令。')
        return '起始位已保存；复位按钮将返回这个姿态'

    def go_home(self, acknowledged, owner):
        with self.lock:
            if not acknowledged:
                raise ValueError('请先勾选从臂执行确认')
            if self.phase in {'executing', 'preview', 'preparing', 'returning'} or self.inflight:
                raise RuntimeError('请先停止预览 / 执行，再复位')
            self._health(cameras=False, enabled=True)
            self._stationary()
            if self.home is None:
                raise RuntimeError('尚未设置起始位，请先点击“记住当前为起始位”')
            motion = Trajectory(self.sample.follower, [self.home], self.config, speed=1.0)
            self._claim(owner)
            self.generation += 1
            self.active_plan = None
            self.home_motion = motion
            self.home_started = self.home_last_tick = time.monotonic()
            self.last_sent = self.sample.follower.copy()
            self.phase = 'returning'
            self.prepared = True
            self.error = ''
            self.last_stop_reason = ''
            self.log(f'复位到起始位：平滑关节轨迹，计划约 {motion.duration:.1f} 秒，可随时停止。')
        return f'正在返回起始位，计划约 {motion.duration:.1f} 秒'

    def _claim(self, owner):
        if not owner:
            raise RuntimeError('请从浏览器页面操作')
        if self.owner and self.owner != owner:
            raise RuntimeError('已有另一个页面控制本次运行；请先停止当前运行')
        self.owner = owner
        self.heartbeat_at = time.monotonic()

    def heartbeat(self, owner):
        with self.lock:
            if owner == self.owner:
                self.heartbeat_at = time.monotonic()

    def connect(self):
        with self.operation_lock:
            with self.lock:
                if self.connected:
                    return '设备已连接'
            robot = SimRobot(self.config) if self.config.mode == 'sim' else PiperHardware(self.config)
            if isinstance(robot, SimRobot):
                robot.demo = False
            cameras = {key: Camera(getattr(self.config, key), wrist=key == 'wrist_image')
                       for key in ('image', 'wrist_image')}
            try:
                robot.connect()
                for camera in cameras.values():
                    camera.connect()
                with self.lock:
                    self.robot, self.cameras = robot, cameras
                    self.connected = True
                    self.error = ''
                    self.last_sent = None
                    self.prepared = False
                    self.log('从臂 CAN 与相机读取已启动；未使能或驱动机械臂。')
            except BaseException:
                for camera in cameras.values():
                    camera.close()
                robot.close()
                raise
        return '设备已连接，等待实时反馈与两路画面'

    def disconnect(self):
        with self.operation_lock:
            with self.lock:
                self._stop_locked('已断开设备通信', pause=False)
                robot, cameras = self.robot, self.cameras
                self.robot, self.cameras = None, {}
                self.connected = False
                self.sample = None
                self.last_sent = None
            if robot:
                robot.close()
            for camera in cameras.values():
                camera.close()
        return '设备通信已断开'

    def check_model(self, url):
        with self.lock:
            if self.phase in {'preview', 'executing', 'preparing', 'returning'} or self.inflight:
                raise RuntimeError('请先停止当前运行')
            generation = self.generation
        client = DemoPolicy() if self.config.mode == 'sim' else PolicyClient(url.strip())
        metadata = client.check()
        with self.lock:
            if generation != self.generation or self.phase in {'preview', 'executing', 'preparing', 'returning'}:
                if hasattr(client, 'close'):
                    client.close()
                raise RuntimeError('运行状态已改变；未切换模型连接')
            previous = self.client
            self.client, self.model = client, metadata
            self.log('模型已连接：' + metadata['model'])
        if hasattr(previous, 'close'):
            previous.close()
        return '模型就绪：' + metadata['model']

    def _health(self, cameras=True, enabled=False):
        s = self.sample
        if not self.connected or s is None:
            raise RuntimeError('请先连接从臂与相机')
        if s.error:
            raise RuntimeError(s.error)
        if s.follower is None or s.feedback_age + time.monotonic() - self.sample_at > self.config.stale_after_s:
            raise RuntimeError('从臂关节 / 夹爪反馈超时')
        if not np.isfinite(s.follower).all():
            raise RuntimeError('从臂反馈包含无效值')
        if self.robot.external_control_recent():
            raise RuntimeError('检测到外部 CAN 控制指令；请关闭其他机械臂控制程序并检查主臂连接')
        if enabled and not s.enabled:
            raise RuntimeError('从臂未使能或已掉使能')
        if cameras:
            stamps = []
            for key, cam in self.cameras.items():
                im, age, stamp = cam.read()
                if im is None or age > self.config.camera_stale_after_s or cam.error:
                    raise RuntimeError(f'{key}：{cam.error or "相机画面超时"}')
                stamps.append(stamp)
            if len(stamps) != 2 or max(stamps) - min(stamps) > 0.2:
                raise RuntimeError('两路相机时间差超过 200 ms')
        return s

    def prepare(self, acknowledged, owner):
        with self.lock:
            if not acknowledged:
                raise ValueError('请先勾选从臂执行确认')
            if self.phase in {'executing', 'preview', 'preparing', 'returning'} or self.inflight:
                raise RuntimeError('请先暂停预览 / 执行，再准备从臂')
            self._health()
            validate_targets([self.sample.follower], self.config)
            self._claim(owner)
            self.phase = 'preparing'
            self.prepared = False
            self.prepare_deadline = time.monotonic() + 3
            self.next_enable = 0
            self.error = ''
            self.log('开始使能并保持当前姿态；不回零、不重置机械零点。')
        return '正在准备从臂；完成后再点击“开始执行”'

    def start(self, mode, task, prompt, duration, chunk_steps, speed, owner, gripper_speed=None):
        task, prompt = str(task).strip(), str(prompt).strip()
        duration, chunk_steps, speed = float(duration), int(chunk_steps), float(speed)
        gripper_speed = self.config.max_gripper_speed_mm_s if gripper_speed is None else float(gripper_speed)
        if not task or not prompt:
            raise ValueError('请填写任务名称与英文提示词')
        if mode not in {'preview', 'executing'} or not ((duration == 0 or 1 <= duration <= 3600) and 1 <= chunk_steps <= 10 and 1 <= speed <= 15):
            raise ValueError('执行参数超出允许范围')
        if not 1 <= gripper_speed <= 100:
            raise ValueError('夹爪速度须在 1–100 mm/s')
        with self.lock:
            if self.phase in {'preview', 'executing', 'preparing', 'returning'} or self.inflight:
                raise RuntimeError('请先暂停当前运行；等待上一个模型请求结束')
            self._health(enabled=mode == 'executing')
            if not self.model:
                raise RuntimeError('请先连接模型服务')
            if mode == 'executing':
                if not self.prepared or self.owner != owner:
                    raise RuntimeError('请先在本页面准备从臂执行')
                self._stationary()
                if self.last_sent is None or np.max(np.abs(self.last_sent[:6] - self.sample.follower[:6])) > 3000:
                    raise RuntimeError('当前位置与保持目标不一致，请重新准备从臂')
                if self.home is None:
                    self._write_home(self.last_sent, '本次首次开始推理前的姿态')
            self._claim(owner)
            self.generation += 1
            self.phase = mode
            self.config.max_joint_speed_deg_s = speed
            self.config.max_gripper_speed_mm_s = gripper_speed
            self.prompt, self.chunk_steps = prompt, chunk_steps
            self.run_started = time.monotonic()
            self.run_ended = 0.0
            self.last_stop_reason = ''
            self.deadline = self.run_started + duration if duration else float('inf')
            self.active_plan = None
            self.inference_deadline = None
            self.next_infer = 0
            self.command_velocity = np.zeros(7)
            self.last_control_send_at = None
            self.missed_control_ticks = 0
            self.sent_count = self.plan_count = 0
            self.prediction = None
            self.error = self.prediction_note = ''
            label = re.sub(r'[^\w\-]+', '_', task)[:60] or 'task'
            self.run_dir = self.run_root / (time.strftime('%Y%m%d-%H%M%S') + '-' + label + '-' + uuid.uuid4().hex[:6])
            self.run_dir.mkdir()
            (self.run_dir / 'meta.json').write_text(json.dumps({'task': task, 'prompt': prompt,
                'mode': mode, 'model': self.model, 'config': self.config.to_dict(), 'chunk_steps': chunk_steps,
                'execution_mode': 'continuous_pchip', 'replan_interval_s': chunk_steps / self.config.fps,
                'command_rate_hz': self.config.control_hz,
                'max_inference_age_s': MAX_INFERENCE_AGE_S, 'plan_buffer_s': PLAN_BUFFER_S,
                'buffer_clock': 'publication; capped by native horizon; never extended',
                'duration_limit': duration, 'state_semantics': 'previous_sent_command; initial=fresh_feedback',
                'action_units': 'millidegree_micrometer'}, ensure_ascii=False, indent=2))
            if self.journal:
                self.journal.close()
            self.journal = (self.run_dir / 'events.jsonl').open('a', buffering=1)
            self._event('start', mode=mode)
            self.log(('开始预测预览' if mode == 'preview' else '开始模型执行') + '：' + task)
        return '已启动预测预览；当前进度见状态卡' if mode == 'preview' else '已启动执行；任务完成后点击停止'

    def _stationary(self):
        now = time.monotonic()
        rows = [(t, q) for t, q in self.trace if now - t < 0.3]
        if not rows or rows[-1][0] - rows[0][0] < 0.15 or np.any(np.ptp([q for _, q in rows], axis=0) > np.array([200] * 6 + [500])):
            raise RuntimeError('请等待机械臂静止，再开始执行')

    def _hold_locked(self):
        if not self.robot or self.sample is None:
            return
        try:
            s = self._health(cameras=False)
            if s.enabled:
                self.last_sent = self.robot.send(s.follower)
                self._event('hold', sent=self.last_sent.tolist())
        except Exception as exc:
            self.log('已停止模型目标发送；当前反馈不足以发送保持指令：' + str(exc))

    def _stop_locked(self, reason, pause=False, fault=False):
        moving = self.phase in {'executing', 'preparing', 'returning'}
        self.generation += 1
        self.phase = 'paused' if pause else 'fault' if fault else 'idle'
        self.active_plan = None
        self.inference_deadline = None
        self.command_velocity = np.zeros(7)
        self.last_control_send_at = None
        self.home_motion = None
        self.last_stop_reason = reason
        if self.run_started and not self.run_ended:
            self.run_ended = time.monotonic()
        self.next_infer = float('inf')
        if moving:
            self._hold_locked()
        if not pause:
            self.prepared = False
            self.owner = None
        if fault:
            self.error = reason
        try:
            self._event('stop', reason=reason, commands=self.sent_count, plans=self.plan_count)
        except OSError as exc:
            self.log('停止已生效；运行日志写入失败：' + str(exc))
        finally:
            if self.journal:
                try:
                    self.journal.close()
                except OSError:
                    pass
                self.journal = None
        self.log(reason)

    def stop(self, pause=False):
        with self.lock:
            self._stop_locked('已暂停，继续时重新观察与预测' if pause else '已停止，旧预测已清空', pause=pause)
        return '已暂停；点击开始可重新预测' if pause else '已停止并清空预测；再次执行请准备从臂'

    def _control_tick(self, now):
        # Caller owns lock. Stop and send share this barrier, so no late result
        # can issue a command after stop returns.
        if self.phase not in {'executing', 'preview', 'preparing', 'returning'}:
            return
        if now - self.heartbeat_at > 3:
            raise RuntimeError('页面心跳中断，已停止；请保持推理页面在前台')
        self._health(cameras=self.phase != 'returning', enabled=self.phase in {'executing', 'returning'})
        if self.phase == 'returning':
            if np.max(np.abs(self.sample.follower[:6] - self.last_sent[:6])) > self.config.max_follow_error_deg * 1000:
                raise RuntimeError('复位时关节跟随误差超限，已停止')
            if now - self.home_started > self.home_motion.duration + 5:
                raise RuntimeError('复位等待实际反馈到位超时，已停止')
            target, _, done = self.home_motion.at(now - self.home_started)
            dt = min(max(now - self.home_last_tick, 0.001), 0.05)
            self.home_last_tick = now
            self.last_sent = self.robot.send(limited_step(self.last_sent, target, dt, self.config))
            error = np.abs(self.sample.follower - self.home)
            arrived = np.all(error <= HOME_TOLERANCE)
            if done and np.max(np.abs(self.last_sent - self.home)) < 3 and arrived:
                self.home_motion = None
                self.phase = 'idle'
                self.last_stop_reason = (f'已回到起始位：最大关节误差 {error[:6].max() / 1000:.2f}°'
                                         '（允许 2.5°），可开始下一轮推理')
                self.log(self.last_stop_reason)
            return
        if self.phase == 'preparing':
            if now >= self.prepare_deadline:
                raise RuntimeError('从臂使能超时，请检查电机状态')
            if self.sample.enabled:
                self.last_sent = self.robot.send(self.sample.follower)
                self.prepared = True
                self.phase = 'idle'
                self.log('从臂已使能并保持当前姿态，可以开始执行。')
            elif now >= self.next_enable:
                if isinstance(self.robot, SimRobot):
                    self.robot.prepare()
                    self.robot.enabled = True
                else:
                    self.robot.request_enable()
                self.next_enable = now + 0.2
            return
        if now >= self.deadline:
            self._stop_locked('达到设定时长，已停止；这不代表任务完成，可增加时长或选择手动停止')
            return
        if self.phase != 'executing':
            return
        # Independent watchdog: a stuck send/recv or planner cannot leave the
        # control thread consuming old motion past the request freshness limit.
        if self.inference_deadline is not None and now > self.inference_deadline:
            raise RuntimeError(INFERENCE_TIMEOUT_MESSAGE)
        if self.last_sent is not None and np.max(np.abs(self.sample.follower[:6] - self.last_sent[:6])) > self.config.max_follow_error_deg * 1000:
            raise RuntimeError('关节跟随误差超限，已停止')
        if self.active_plan is None:
            return
        period = 1 / self.config.control_hz
        dt = period if self.last_control_send_at is None else now - self.last_control_send_at
        if dt < period * .5:
            return  # Duplicate ticks cannot burst commands.
        if dt > MAX_CONTROL_GAP_S:
            raise RuntimeError('控制周期超过 100 ms，已停止；不突发补发积压动作')
        target, model_index, old_weight = self.active_plan.at(now)
        command, velocity = governed_step(self.last_sent, target, self.command_velocity, dt, self.config)
        sent = self.robot.send(command)
        self.last_sent = np.asarray(sent, dtype=np.float32)
        self.command_velocity = velocity
        self.last_control_send_at = now
        self.sent_count += 1
        self._event('command', sent=self.last_sent.tolist(), prediction=target.tolist(),
                    feedback=self.sample.follower.tolist(), index=model_index,
                    plan_number=self.active_plan.number,
                    previous_plan_number=self.active_plan.previous_number, previous_weight=old_weight,
                    control_dt_s=dt, monotonic_time=now)

    def _control_loop(self):
        # Deadline scheduling: include time spent reading/sending in the cycle,
        # rather than adding another sleep after every CAN operation.
        period = 1 / self.config.control_hz
        next_tick = time.monotonic()
        while not self.shutdown.is_set():
            if self.shutdown.wait(max(0, next_tick - time.monotonic())):
                break
            try:
                with self.lock:
                    if self.robot:
                        self.sample = self.robot.read()
                        self.sample_at = time.monotonic()
                        if self.sample.follower is not None:
                            self.trace.append((self.sample_at, self.sample.follower.copy()))
                        self._control_tick(self.sample_at)
            except Exception as exc:
                with self.lock:
                    if self.phase in {'preview', 'executing', 'preparing', 'returning'}:
                        self._stop_locked(str(exc), fault=True)
                    elif self.error != str(exc):
                        self.error = str(exc)
                        self.log(str(exc))
            next_tick += period
            finished = time.monotonic()
            if next_tick < finished:
                skipped = int((finished - next_tick) / period) + 1
                next_tick += skipped * period
                with self.lock:
                    if self.phase == 'executing':
                        self.missed_control_ticks += skipped

    def _planner_loop(self):
        while not self.shutdown.wait(0.02):
            with self.lock:
                if self.phase not in {'preview', 'executing'} or time.monotonic() < self.next_infer:
                    continue
                generation, mode, run = self.generation, self.phase, self.run_dir
                try:
                    self._health(enabled=mode == 'executing')
                    observed_at = time.monotonic()
                    state = self.last_sent if self.last_sent is not None else self.sample.follower
                    obs = {'state': state.astype(np.float32).copy(), 'prompt': self.prompt}
                    camera_stamps = {}
                    for key, camera in self.cameras.items():
                        im, _, stamp = camera.read()
                        obs[key] = im.copy()
                        camera_stamps[key] = stamp
                    self.inflight = True
                    self.inference_deadline = observed_at + MAX_INFERENCE_AGE_S if mode == 'executing' else None
                    self._event('inference_request', number=self.plan_count + 1,
                                observed_at=observed_at, deadline_at=self.inference_deadline)
                    client = self.client
                except Exception as exc:
                    self._stop_locked(str(exc), fault=True)
                    continue
            try:
                began = time.monotonic()
                try:
                    actions, timing = client.infer(obs, timeout=MAX_INFERENCE_AGE_S if mode == 'executing' else 5)
                except TimeoutError as exc:
                    raise RuntimeError(INFERENCE_TIMEOUT_MESSAGE if mode == 'executing'
                                       else '预览预测等待超时，请检查模型服务和网络') from exc
                latency = time.monotonic() - began
                if mode == 'executing' and time.monotonic() - observed_at > MAX_INFERENCE_AGE_S:
                    raise RuntimeError(INFERENCE_TIMEOUT_MESSAGE)
                if actions.shape != (50, 7) or not np.isfinite(actions).all():
                    raise ValueError('模型输出不是有效的 50 × 7 动作')
                note = ''
                blocked = ''
                try:
                    checked, note = bounded_prediction(actions, obs['state'], self.config)
                except ValueError as exc:
                    checked = None
                    blocked = str(exc)
                    note = '禁止执行：' + blocked
                with self.lock:
                    if generation != self.generation or self.phase != mode:
                        continue
                    self._health(enabled=mode == 'executing')
                    self.prediction, self.prediction_at = actions.copy(), time.monotonic()
                    self.prediction_note = note
                    self.latency_ms, self.server_ms = latency * 1000, timing.get('infer_ms')
                    self.plan_count += 1
                    number = self.plan_count
                    stream = None
                    if mode == 'executing' and not blocked:
                        stream = make_stream_plan(checked, obs['state'], self.last_sent,
                            observed_at=observed_at, published_at=time.monotonic(), number=number,
                            config=self.config, previous=self.active_plan)
                    self._event('prediction', state=obs['state'].tolist(), actions=actions.tolist(),
                                latency_ms=self.latency_ms, server_ms=self.server_ms, number=number,
                                camera_stamps=camera_stamps, observed_at=observed_at, validation=note,
                                executable=not bool(blocked), transport=timing,
                                stream=stream.summary() if stream else None)
                    if mode == 'executing':
                        if blocked:
                            raise RuntimeError(blocked)
                        self.active_plan = stream
                        # Predict from the next fresh observation while this
                        # plan continues to play. No queued observation backlog.
                        self.next_infer = next_observation_at(stream, self.chunk_steps / self.config.fps)
                    else:
                        self.next_infer = time.monotonic() + 0.25
                    self.inference_deadline = None
                # Archive the exact input views, outside the control lock.
                for key in ('image', 'wrist_image'):
                    Image.fromarray(obs[key]).save(run / f'{number:05d}-{key}.jpg', quality=85)
            except Exception as exc:
                with self.lock:
                    if generation == self.generation and self.phase == mode:
                        self._stop_locked(str(exc), fault=True)
            finally:
                with self.lock:
                    self.inflight = False
                    self.inference_deadline = None

    def snapshot(self, owner=None):
        if owner:
            self.heartbeat(owner)
        with self.lock:
            elapsed = ((self.run_ended or time.monotonic()) - self.run_started) if self.run_started else 0.0
            issues = []
            try:
                self._health()
            except Exception as exc:
                issues.append(str(exc))
            return {'phase': self.phase, 'connected': self.connected, 'prepared': self.prepared,
                    'sample': self.sample, 'images': {k: c.read()[0] for k, c in self.cameras.items()},
                    'camera_ages': {k: c.read()[1] for k, c in self.cameras.items()},
                    'model': dict(self.model), 'prediction': self.prediction,
                    'prediction_note': self.prediction_note, 'prediction_at': self.prediction_at,
                    'last_sent': self.last_sent, 'latency_ms': self.latency_ms, 'server_ms': self.server_ms,
                    'plans': self.plan_count, 'commands': self.sent_count, 'inflight': self.inflight,
                    'issues': issues, 'error': self.error, 'logs': '\n'.join(self.logs),
                    'trace': [q for _, q in self.trace], 'remaining': max(0, self.deadline - time.monotonic()) if np.isfinite(self.deadline) else None,
                    'home': self.home.copy() if self.home is not None else None, 'home_source': self.home_source,
                    'elapsed': elapsed, 'command_hz': self.sent_count / max(elapsed, 0.01),
                    'control_rate_hz': self.config.control_hz,
                    'missed_control_ticks': self.missed_control_ticks,
                    'buffer_remaining_s': max(0, self.active_plan.expires_at - time.monotonic()) if self.active_plan else 0,
                    'inference_wait_s': max(0, time.monotonic() - (self.inference_deadline - MAX_INFERENCE_AGE_S)) if self.inference_deadline is not None else None,
                    'last_stop_reason': self.last_stop_reason,
                    'run': str(self.run_dir) if self.run_dir else ''}

    def runs(self):
        result = []
        for p in sorted(self.run_root.glob('*/meta.json'), reverse=True):
            meta = json.loads(p.read_text())
            result.append((p.parent.name + (' · 仅预览' if meta['mode'] == 'preview' else ' · 执行'), p.parent.name))
        return result

    def replay(self, name, index=0):
        root = (self.run_root / str(name)).resolve()
        if root.parent != self.run_root:
            raise ValueError('运行记录不存在')
        records = [json.loads(s) for s in (root / 'events.jsonl').read_text().splitlines()]
        predictions = [s for s in records if s['event'] == 'prediction']
        if not predictions:
            raise ValueError('此运行尚未产生预测')
        index = min(max(int(index), 0), len(predictions) - 1)
        row = predictions[index]
        end = predictions[index + 1]['time'] if index + 1 < len(predictions) else float('inf')
        row['sent_commands'] = [r for r in records if r['event'] == 'command' and
            (r['plan_number'] == row['number'] if 'plan_number' in r else row['time'] <= r['time'] < end)]
        images = []
        for key in ('image', 'wrist_image'):
            p = root / f'{row["number"]:05d}-{key}.jpg'
            images.append(np.asarray(Image.open(p)) if p.exists() else None)
        return row, images, len(predictions), index

    def close(self):
        self.stop()
        self.shutdown.set()
        self.worker.join(timeout=2)
        self.planner.join(timeout=6)
        self.disconnect()
        if hasattr(self.client, 'close'):
            self.client.close()
