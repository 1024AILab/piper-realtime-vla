"""Loopback-only, warmed OpenPI service for Piper Studio. No hardware I/O."""
import io
import json
import logging
from pathlib import Path
import resource
import time

import jax
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from openpi.policies import policy_config
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.training import config

BASE = Path('/root/autodl-tmp/piper-training')
CHECKPOINT = BASE / 'checkpoints/piper_duck_lora/duck_en_v1/29999'
PROMPT = 'Pick up the yellow duck and place it in the pen holder.'


class ImageTransportPolicy:
    """Decode lossless RGB transport before the unchanged trained policy."""
    def __init__(self, policy):
        self.policy = policy

    def infer(self, observation):
        observation = dict(observation)
        encoding = observation.pop('image_encoding', 'numpy')
        if encoding == 'png':
            for key in ('image', 'wrist_image'):
                data = observation[key]
                if not isinstance(data, bytes) or len(data) > 2 * 1024 * 1024:
                    raise ValueError('Invalid PNG payload')
                with Image.open(io.BytesIO(data)) as im:
                    if im.format != 'PNG' or im.size != (640, 480) or im.mode != 'RGB':
                        raise ValueError('Expected lossless 640x480 RGB PNG')
                    observation[key] = np.asarray(im).copy()
        elif encoding != 'numpy':
            raise ValueError('Unsupported image encoding')
        return self.policy.infer(observation)


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    logging.basicConfig(level=logging.INFO)
    device = jax.devices('gpu')[0]
    jax.block_until_ready(jax.numpy.zeros(1))
    limit = device.memory_stats()['bytes_limit']
    if not 7 * 2**30 < limit <= 21 * 2**30:
        raise RuntimeError('请先 source inference_env.sh；JAX 显存池上限不符合并行运行配置')
    cfg = config.get_config('piper_duck_lora')
    print('Loading pi0.5 LoRA checkpoint 29999', flush=True)
    policy = policy_config.create_trained_policy(cfg, CHECKPOINT)
    frame = pq.ParquetFile('/root/autodl-tmp/duck/data/chunk-000/episode_000000.parquet').read_row_group(0).slice(0, 1).to_pylist()[0]
    obs = {'state': np.asarray(frame['state'], dtype=np.float32), 'prompt': PROMPT}
    for key in ('image', 'wrist_image'):
        obs[key] = np.asarray(Image.open(io.BytesIO(frame[key]['bytes'])).convert('RGB'))
    started = time.monotonic()
    actions = np.asarray(policy.infer(obs)['actions'])
    if actions.shape != (50, 7) or not np.isfinite(actions).all():
        raise RuntimeError('Warmup produced invalid actions')
    metadata = dict(cfg.policy_metadata or {})
    metadata.update(studio_protocol=1, model='pi0.5 LoRA / duck / 29999', action_horizon=50,
                    action_dim=7, action_units='millidegree_micrometer', action_semantics='absolute',
                    state_semantics='previous_command', fps=10, checkpoint=str(CHECKPOINT),
                    prompt=PROMPT, warmup_seconds=time.monotonic() - started,
                    image_encodings=['numpy', 'png'])
    print('READY ' + json.dumps(metadata), flush=True)
    WebsocketPolicyServer(ImageTransportPolicy(policy), host='127.0.0.1', port=8000, metadata=metadata).serve_forever()


if __name__ == '__main__':
    main()
