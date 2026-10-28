import io

import msgpack
import numpy as np
import pytest
from PIL import Image

from piper_studio import policy_client as module


META = {'studio_protocol': 1, 'action_horizon': 50, 'action_dim': 7,
        'action_units': 'millidegree_micrometer', 'action_semantics': 'absolute',
        'state_semantics': 'previous_command', 'fps': 10,
        'image_encodings': ['numpy', 'png']}


def observation():
    rng = np.random.default_rng(5)
    return {'state': np.arange(7, dtype=np.float32) * 1000, 'prompt': 'pick up the duck',
            'image': rng.integers(0, 256, (480, 640, 3), dtype=np.uint8),
            'wrist_image': rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)}


def test_png_preserves_both_rgb_images_and_raw_state_exactly():
    obs = observation()
    wire = module.unpack(module.encode_observation(obs, png=True))
    assert wire['image_encoding'] == 'png'
    assert wire['prompt'] == obs['prompt']
    np.testing.assert_array_equal(wire['state'], obs['state'])
    for key in ('image', 'wrist_image'):
        with Image.open(io.BytesIO(wire[key])) as im:
            assert im.mode == 'RGB' and im.size == (640, 480)
            np.testing.assert_array_equal(np.asarray(im), obs[key])
        assert isinstance(obs[key], np.ndarray)


class SocketStub:
    def __init__(self, metadata=None):
        self.metadata = dict(META if metadata is None else metadata)
        self.requests = []
        self.greeted = self.closed = self.fail = False

    def recv(self, timeout):
        if not self.greeted:
            self.greeted = True
            return msgpack.packb(self.metadata)
        if self.fail:
            raise TimeoutError('simulated network timeout')
        return msgpack.packb({'actions': np.zeros((50, 7), np.float32),
                             'server_timing': {'infer_ms': 150}}, default=module.encode_array)

    def send(self, payload):
        self.requests.append(module.unpack(payload))

    def close(self):
        self.closed = True


def test_client_reuses_connection_but_does_not_retry_failed_observation(monkeypatch):
    sockets = []
    def connect(*args, **kwargs):
        sock = SocketStub()
        sockets.append(sock)
        return sock
    monkeypatch.setattr(module, 'connect', connect)
    client = module.PolicyClient()  # Negotiate lossless PNG by default.
    client.check()
    obs = observation()
    for _ in range(2):
        actions, timing = client.infer(obs)
        assert actions.shape == (50, 7)
        assert timing['image_encoding'] == 'png'
        assert timing['payload_bytes'] > 0
    assert len(sockets) == 1 and len(sockets[0].requests) == 2
    sockets[0].fail = True
    with pytest.raises(TimeoutError):
        client.infer(obs)
    assert sockets[0].closed and len(sockets) == 1
    assert len(sockets[0].requests) == 3  # Exactly one send; no replay after timeout.
    client.infer(obs)  # A new explicit request reconnects.
    assert len(sockets) == 2
    client.close()
    assert sockets[1].closed


def test_client_negotiates_original_numpy_transport(monkeypatch):
    metadata = dict(META)
    del metadata['image_encodings']
    sock = SocketStub(metadata)
    monkeypatch.setattr(module, 'connect', lambda *a, **k: sock)
    client = module.PolicyClient()
    obs = observation()
    _, timing = client.infer(obs)
    assert timing['image_encoding'] == 'numpy'
    assert 'image_encoding' not in sock.requests[0]
    np.testing.assert_array_equal(sock.requests[0]['image'], obs['image'])
    client.close()


def test_wrong_model_metadata_is_rejected_and_connection_closed(monkeypatch):
    metadata = dict(META, action_units='radians')
    sock = SocketStub(metadata)
    monkeypatch.setattr(module, 'connect', lambda *a, **k: sock)
    client = module.PolicyClient()
    with pytest.raises(RuntimeError, match='不匹配'):
        client.check()
    assert sock.closed and not sock.requests
