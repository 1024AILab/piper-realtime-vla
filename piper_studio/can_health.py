"""Decode Linux/gs_usb CAN error *and state* notifications.

Reference: https://github.com/torvalds/linux/blob/master/include/uapi/linux/can/error.h
Recovery never resumes a recording or sends any bus command.
"""
from __future__ import annotations

import collections
import threading
import time


def decode_can_event(mask, data):
    mask &= 0x1FFFFFFF
    raw = bytes(data)
    detail = {"mask": hex(mask), "data": raw.hex(" "), "recovery": False,
              "blocking": True, "needs_active": False}
    if len(raw) != 8:
        return {**detail, "message": f"CAN 状态帧长度异常：{len(raw)} 字节", "needs_active": True}
    causes = []
    controller = raw[1] if mask & 0x04 else 0
    for bit, label in ((0x01, "发送超时"), (0x08, "协议错误"), (0x10, "收发器异常"),
                       (0x20, "发送未收到 ACK"), (0x40, "控制器 Bus-off"), (0x80, "总线错误")):
        if mask & bit:
            causes.append(label)
    for bit, label in ((0x01, "接收缓冲区溢出"), (0x02, "发送缓冲区溢出"),
                       (0x04, "接收错误计数达到警告级别"), (0x08, "发送错误计数达到警告级别"),
                       (0x10, "接收进入 Error-passive"), (0x20, "发送进入 Error-passive")):
        if controller & bit:
            causes.append(label)
    if mask & 0x04 and (controller == 0 or controller & 0x80):
        causes.append("未识别的控制器状态")
        detail["needs_active"] = True
    if mask & 0x200:
        detail.update(tx_error_count=raw[6], rx_error_count=raw[7])
        if max(raw[6:8]) >= 96 and not causes:
            causes.append("错误计数达到警告级别")
        if max(raw[6:8]) >= 128:
            detail["needs_active"] = True
    if mask & ~0x3FF or mask == 0:
        causes.append("未识别的 CAN 错误状态")
        detail["needs_active"] = True
    if mask & 0x40 or controller & 0x30:
        detail["needs_active"] = True
    if causes:
        detail["message"] = "CAN：" + "；".join(causes)
    elif controller & 0x40:
        detail.update(blocking=False, recovery=True, message="CAN 控制器恢复 Error-active 正常状态")
    elif mask & 0x100:
        # A controller restart is recovery, but may have interrupted the stream.
        detail.update(recovery=True, message="CAN 控制器已重启，等待收包稳定")
    elif mask & 0x02:
        detail.update(blocking=False, message="CAN 仲裁让出（总线正常竞争）")
    else:
        detail.update(blocking=False, message="CAN 错误计数状态更新")
    return detail


class CanHealth:
    """Transient faults require a quiet interval plus continuous valid reception.

    Bus-off, Error-passive and unknown states additionally need an explicit
    active/restarted notification. Time alone can never clear a fault.
    """
    def __init__(self):
        self.lock = threading.Lock()
        self.received_frames = 0
        self.error_frames = 0
        self.status_frames = 0
        self.events = collections.deque(maxlen=32)
        self.fault = ""
        self.last_fault_at = None
        self.needs_active = False
        self.last_rx = None
        self.clean_since = None
        self.clean_frames = 0
        self.recovered = False

    def _event(self, item, now):
        # Keep exact counters without flooding disk/UI on repeated error frames.
        if self.events and self.events[-1]["message"] == item["message"] and now - self.events[-1]["at"] < 1:
            self.events[-1]["repeats"] += 1
            return None
        event = {**item, "at": now, "time": time.strftime("%H:%M:%S"), "repeats": 1}
        self.events.append(event)
        return event.copy()

    def feed_error(self, mask, data, now=None):
        now = time.monotonic() if now is None else now
        item = decode_can_event(mask, data)
        with self.lock:
            if item["blocking"] and not item["recovery"]:
                self.error_frames += 1
            else:
                self.status_frames += 1
            if item["blocking"]:
                self.fault = item["message"]
                self.last_fault_at = now
                self.clean_since = None
                self.clean_frames = 0
                self.recovered = False
                self.needs_active = item["needs_active"] or (self.needs_active and not item["recovery"])
            else:
                if item["recovery"]:
                    self.needs_active = False
            return self._event(item, now)

    def feed_valid(self, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            self.received_frames += 1
            if self.clean_since is None or self.last_rx is None or now - self.last_rx > 0.25:
                self.clean_since, self.clean_frames = now, 0
            self.clean_frames += 1
            self.last_rx = now

    def snapshot(self, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            if (self.fault and not self.needs_active and self.last_fault_at is not None
                    and now - self.last_fault_at >= 1 and self.clean_since is not None
                    and now - self.clean_since >= 1 and self.clean_frames >= 20
                    and self.last_rx is not None and now - self.last_rx <= 0.25):
                self.fault = ""
                self.recovered = True
                self._event({"message": "CAN 接收已恢复：连续正常收包至少 1 秒；被中断的录制仍需人工处理"}, now)
            stale = self.last_rx is not None and now - self.last_rx > 0.5
            state = ("异常" if self.fault else "无新数据" if stale else "已恢复" if self.recovered
                     else "正常收包" if self.last_rx is not None else "等待收包")
            return {"state": state, "error": self.fault,
                    "received_frames": self.received_frames, "error_frames": self.error_frames,
                    "status_frames": self.status_frames, "needs_active_notification": self.needs_active,
                    "last_rx_age_s": now - self.last_rx if self.last_rx is not None else None,
                    "recent_events": [e.copy() for e in self.events]}
