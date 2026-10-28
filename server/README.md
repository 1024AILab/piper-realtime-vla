# OpenPI service adapter

This service runs inside your existing OpenPI GPU environment. It does not
install OpenPI, download weights, train a model, or access robot hardware.

The archived deployment used OpenPI commit
`215abfb217dbac7d5f1273282331b9b1866c0479`. Keep the exact checkpoint,
normalization assets, tokenizer, data transforms and model configuration from
your training environment. A stock LIBERO config is not interchangeable with
the Piper config.

```bash
python /path/to/piper-realtime-vla/server/serve_openpi.py \
  --config piper_duck_lora \
  --checkpoint /path/to/checkpoint \
  --warmup-parquet /path/to/episode_000000.parquet
```

If your config is not in OpenPI's registry, use
`--config-factory your_adapter:make_config` instead of `--config`.
The installed module must be importable in that environment.

The protocol requires two 640×480 RGB images, a seven-value state containing
the previous command, and **absolute** `(50, 7)` targets at 10 Hz. The first six
values use 0.001 degrees; the gripper uses micrometers. The adapter checks shape
and finite values. The operator must verify the checkpoint's action semantics
and units; metadata cannot infer them from weights.

Forward the loopback service to the client computer:

```bash
ssh -N -L 127.0.0.1:8000:127.0.0.1:8000 your-gpu-host
```

The reference directory preserves the original deployment scripts and training
configuration as migration references. Their paths and GPU-memory thresholds
are specific to the original deployment; they are not the portable launch path.
`piper_duck_config.py` also depends on the external
`processer.libero_policy` adapter from
[ROBOT_DATASET](https://github.com/MAZHIPENG001/ROBOT_DATASET), commit
`30ec82bcc7b8e4636d5838da1a19330dc74e5da2`.
That external code and checkpoint weights are not bundled here.

## 中文

在训练该权重的 OpenPI GPU 环境中运行上面的命令，保留对应的数据变换、
归一化统计、tokenizer 和模型配置。`--config` 使用已注册的配置名称；
`--config-factory 模块:函数` 可直接返回同一训练配置。

模型必须输出 10 Hz 的 50×7 **绝对目标**，六轴单位为 0.001 度，夹爪为微米。
两路输入均为 640×480 RGB，state 表示上一条已执行指令。服务会验证形状和有限值，
但动作语义与单位仍须与训练配置核对。服务仅监听本机，通过 SSH 隧道连接。
`reference/` 是原部署记录，路径、显存阈值和外部适配依赖需要按环境调整。
