"""Serve a trained, compatible OpenPI policy over a loopback WebSocket.

This adapter has no robot or camera access. Install it in the same GPU
environment used to train the checkpoint, including its data transforms.
"""
import argparse
import importlib
import io
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image


class ImageTransportPolicy:
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
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--config', help='Name registered in the training OpenPI installation')
    selection.add_argument('--config-factory', help='Installed module:function returning the exact training config')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--warmup-parquet', type=Path, required=True)
    parser.add_argument('--prompt', default='Pick up the yellow duck and place it in the pen holder.')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()

    import pyarrow.parquet as pq
    from openpi.policies import policy_config
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from openpi.training import config

    if args.config_factory:
        module, function = args.config_factory.split(':', 1)
        cfg = getattr(importlib.import_module(module), function)()
    else:
        cfg = config.get_config(args.config)
    policy = policy_config.create_trained_policy(cfg, args.checkpoint)
    frame = next(pq.ParquetFile(args.warmup_parquet).iter_batches(batch_size=1)).to_pylist()[0]
    obs = {'state': np.asarray(frame['state'], dtype=np.float32), 'prompt': args.prompt}
    if obs['state'].shape != (7,) or not np.isfinite(obs['state']).all():
        raise ValueError('Warmup state must have seven finite values')
    for key in ('image', 'wrist_image'):
        with Image.open(io.BytesIO(frame[key]['bytes'])) as im:
            obs[key] = np.asarray(im.convert('RGB')).copy()
        if obs[key].shape != (480, 640, 3):
            raise ValueError('Warmup images must be 640x480 RGB')
    started = time.monotonic()
    actions = np.asarray(policy.infer(obs)['actions'])
    if actions.shape != (50, 7) or not np.isfinite(actions).all():
        raise ValueError('This client requires finite absolute actions of shape (50, 7)')
    metadata = dict(cfg.policy_metadata or {})
    metadata.update(studio_protocol=1, model=args.checkpoint.name, action_horizon=50,
                    action_dim=7, action_units='millidegree_micrometer',
                    action_semantics='absolute', state_semantics='previous_command',
                    fps=10, prompt=args.prompt, warmup_seconds=time.monotonic() - started,
                    image_encodings=['numpy', 'png'])
    print('READY ' + json.dumps(metadata), flush=True)
    WebsocketPolicyServer(ImageTransportPolicy(policy), host='127.0.0.1',
                          port=args.port, metadata=metadata).serve_forever()


if __name__ == '__main__':
    main()
