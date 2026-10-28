"""Bounded OpenPI WebSocket client; never imports or accesses robot hardware."""
from __future__ import annotations

import io
import threading
import time
from urllib.parse import urlparse

import msgpack
import numpy as np
from PIL import Image
from websockets.sync.client import connect


def encode_array(value):
    # OpenPI msgpack_numpy wire schema (Physical-Intelligence/openpi).
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in 'buif':
            raise ValueError('Unsupported array dtype')
        return {b'__ndarray__': True, b'data': value.tobytes(),
                b'dtype': value.dtype.str, b'shape': value.shape}
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def decode_array(value):
    if b'__ndarray__' in value:
        dtype = np.dtype(value[b'dtype'])
        shape = value[b'shape']
        if dtype.kind not in 'buif' or len(shape) > 4 or any(n < 0 for n in shape):
            raise ValueError('Invalid array type or shape')
        return np.frombuffer(value[b'data'], dtype=dtype).reshape(shape).copy()
    if b'__npgeneric__' in value:
        dtype = np.dtype(value[b'dtype'])
        if dtype.kind not in 'buif':
            raise ValueError('Invalid scalar type')
        return dtype.type(value[b'data'])
    return value


def unpack(payload):
    if isinstance(payload, str):
        raise RuntimeError('模型服务返回错误：' + payload[-1500:])
    return msgpack.unpackb(payload, object_hook=decode_array)


def encode_observation(observation, png=False):
    wire = dict(observation)
    if png:
        for key in ('image', 'wrist_image'):
            a = np.asarray(observation[key])
            if a.dtype != np.uint8 or a.shape != (480, 640, 3):
                raise ValueError('模型相机输入必须为 640 × 480 uint8 RGB')
            buffer = io.BytesIO()
            Image.fromarray(a).save(buffer, format='PNG', compress_level=1)
            wire[key] = buffer.getvalue()
        wire['image_encoding'] = 'png'
    return msgpack.packb(wire, default=encode_array)


class PolicyClient:
    def __init__(self, url='ws://127.0.0.1:8000', image_encoding='png'):
        parsed = urlparse(url)
        if parsed.scheme != 'ws' or parsed.hostname not in {'127.0.0.1', 'localhost', '::1'} or parsed.username:
            raise ValueError('请使用本机 SSH 隧道地址，例如 ws://127.0.0.1:8000')
        self.url = url
        if image_encoding not in {'numpy', 'png'}:
            raise ValueError('图像传输格式必须为 numpy 或 png')
        # Live D435 comparison on the current Mac -> AutoDL link: PNG halves the
        # payload and is faster. It preserves all pixels and falls back to raw
        # RGB when the server does not advertise this lossless transport.
        self.image_encoding = image_encoding
        self.metadata = {}
        self._socket = None
        self._request_lock = threading.Lock()

    def _open(self, timeout):
        ws = connect(self.url, open_timeout=timeout, close_timeout=0.1, compression=None,
                     max_size=4 * 1024 * 1024, proxy=None)
        try:
            self.metadata = unpack(ws.recv(timeout=timeout))
            expected = {'studio_protocol': 1, 'action_horizon': 50, 'action_dim': 7,
                        'action_units': 'millidegree_micrometer', 'action_semantics': 'absolute',
                        'state_semantics': 'previous_command', 'fps': 10}
            if any(self.metadata.get(k) != v for k, v in expected.items()):
                raise RuntimeError('模型接口不匹配；请启动专用 serve_piper_studio.py')
        except BaseException:
            ws.close()
            raise
        return ws

    def check(self):
        with self._request_lock:
            if self._socket is None:
                self._socket = self._open(3)
            return dict(self.metadata)

    def infer(self, observation, timeout=2):
        start = time.monotonic()
        with self._request_lock:
            try:
                if self._socket is None:
                    self._socket = self._open(timeout)
                encoded_at = time.monotonic()
                use_png = self.image_encoding == 'png' and 'png' in self.metadata.get('image_encodings', [])
                payload = encode_observation(observation, use_png)
                encode_ms = (time.monotonic() - encoded_at) * 1000
                self._socket.send(payload)
                result = unpack(self._socket.recv(timeout=max(0.01, timeout - (time.monotonic() - start))))
            except BaseException:
                # No automatic replay of failed observations. A new operator
                # request may establish a fresh socket after the run stops.
                if self._socket:
                    self._socket.close()
                self._socket = None
                raise
        actions = np.asarray(result['actions'], dtype=np.float32)
        if actions.shape != (50, 7) or not np.isfinite(actions).all():
            raise ValueError('模型动作必须是有限的 50 × 7 数组')
        timing = dict(result.get('server_timing', {}))
        timing.update(encode_ms=encode_ms, payload_bytes=len(payload),
                      image_encoding='png' if use_png else 'numpy')
        return actions, timing

    def close(self):
        with self._request_lock:
            if self._socket:
                self._socket.close()
                self._socket = None


class DemoPolicy:
    """Explicitly labelled UI simulation. Cannot be selected for real hardware."""
    metadata = {'model': '模拟预测 · 不使用训练权重', 'studio_protocol': 1}

    def check(self):
        return self.metadata.copy()

    def infer(self, observation, timeout=2):
        time.sleep(0.05)
        a = np.repeat(np.asarray(observation['state'])[None], 50, axis=0)
        a[:, 0] += np.sin(np.arange(50) / 25) * 2000
        return a, {'infer_ms': 50.0}
