# Piper Realtime VLA

**[English](README.md) | [简体中文](README.zh-CN.md)**

面向松灵 AgileX PiPER 的异步 VLA 执行框架。后台请求模型，独立控制线程持续执行
仍有效的轨迹；通过时间对齐、轨迹衔接和 PCHIP 插值，将 10 Hz 动作块转换为
连续的 50 Hz 控制指令。

![平台 UI](docs/images/platform-ui.jpg)

**平台 UI。** 真实运行的推理界面，截图使用模拟设备与演示策略，不代表训练模型的效果。

数据采集与数据集回放见 [Piper Studio](https://github.com/1024AILab/piper-studio)。


## 真机平台

![Real robot setup / 真机平台](docs/images/real-robot-setup.png)

用户提供的 PiPER 真机平台照片：小黄鸭抓取与笔筒放置场景。


## 推理与执行加速方法

- **异步推理：** 模型请求在后台进行，不阻塞独立控制线程。
- **复用 WebSocket：** 保持模型连接，不自动重放失败的请求。
- **无损 PNG 传输：** 服务端支持时压缩两路 RGB，保留原始像素。
- **连续轨迹：** 用单调时钟对齐动作块，保留短段有效旧轨迹，平滑过渡并用 PCHIP 升频。
- **受限执行：** 保留位置、速度、加速度、数据新鲜度和超时检查。

这些改进减少通信开销与“走一段、停下来等预测”的等待，不改变神经网络本身的计算速度。
默认 **50 Hz** 是控制目标频率，不是操作系统硬实时调度保证。

```mermaid
flowchart LR
  A[RGB 与机械臂状态] --> B[后台 WebSocket 推理]
  B --> C[时间对齐的轨迹缓存]
  C --> D[PCHIP 与速度/加速度约束]
  D --> E[50 Hz CAN 控制线程]
  E --> A
```

## 启动平台 UI

推荐 **Python 3.12**。两个仓库保留了共用的 `piper_studio` 设备模块名称，
请分别创建虚拟环境，不要将两个包安装到同一个环境。

```bash
git clone https://github.com/1024AILab/piper-realtime-vla.git
cd piper-realtime-vla
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m piper_studio.inference_app --sim --connect --port 7862 --open
```

打开 **http://127.0.0.1:7862**，点击“仅预览预测”查看演示动作块。
模拟模式有明确标记，当前应用界面为中文。

## 接入训练好的策略

客户端使用固定的接口约定。其他 VLA 模型需要提供保持这些语义的适配器，
不能直接连接任意模型服务。

| 字段 | 要求 |
|---|---|
| 图像 | `image`、`wrist_image`，640×480 uint8 RGB |
| 状态 | 七维，表示上一条实际执行的目标指令 |
| 输出 | 形状 `(50, 7)`，有限数值的绝对目标 |
| 单位 | 六轴为 0.001 度，夹爪为微米 |
| 动作间隔 | 10 Hz |
| 通信 | OpenPI MessagePack WebSocket，含 `studio_protocol=1` 元数据 |

仓库附带 [OpenPI 服务端适配器](server/README.md)。在原训练 GPU 环境中运行，
使用匹配的权重、归一化统计、tokenizer 和数据变换；仓库不包含模型权重和训练数据集。

```bash
# GPU 服务器：激活训练该权重的 OpenPI 环境
python /path/to/piper-realtime-vla/server/serve_openpi.py \
  --config piper_duck_lora \
  --checkpoint /path/to/checkpoint \
  --warmup-parquet /path/to/episode_000000.parquet

# 连接机械臂的电脑
ssh -N -L 127.0.0.1:8000:127.0.0.1:8000 your-gpu-host
```

服务和网页都只监听本机。在页面中连接 `ws://127.0.0.1:8000`。
`server/reference/` 保留原部署脚本，复用前需要修改机器相关路径与显存参数。

## 连接真机

```bash
python -m pip install -e '.[mac]'
cp configs/mac_inference.json configs/mac_inference.local.json
```

将本地配置中的相机 USB 编号和 CAN 适配器改为真实设备。
设置 `PIPER_SCENE_SERIAL`、`PIPER_WRIST_SERIAL` 后启动 RGB 辅助进程，再启动界面：

```bash
./启动RealSense彩色流.command
# 另开已激活 .venv 的终端，并保持 RGB 取图运行
python -m piper_studio.inference_app --config configs/mac_inference.local.json --connect --open
```

真机页面地址为 **http://127.0.0.1:7861**。连接设备与模型，先检查预测，
然后在页面中准备从臂并执行。主臂须停止向同一 CAN 总线发送控制指令，
现场应可随时操作硬件急停。启动 UI 不会自动准备机械臂运动。

Linux 使用 `.[hardware,realsense]`，配置 SocketCAN 与相机编号。
同一个适配器仅允许一个进程使用，切换前先断开数采平台。

## 默认执行参数

下表对应 `configs/mac_inference.json`；模拟模式的夹爪速度上限为 30 mm/s。

| 参数 | 默认值 |
|---|---|
| 控制频率 / 策略动作频率 | 50 Hz / 10 Hz |
| 重新观察间隔 | 0.5 秒，缓存不足时提前请求 |
| 推理结果年龄上限 | 1.5 秒 |
| 结果就绪后的未来缓存 | 最多 2 秒，受原预测长度限制 |
| 轨迹衔接 | 约 0.1 秒保留段 + 0.3 秒过渡 |
| 关节 / 夹爪速度 | 5°/s / 50 mm/s |
| 关节 / 夹爪加速度 | 25°/s² / 250 mm/s² |
| 夹爪范围 | 0–70 mm |

过期轨迹不会自动延长，晚到的响应不能重启已经停止的运动。
控制器还检查页面心跳、CAN 故障、反馈/图像新鲜度、跟随误差和控制周期超时。
暂停/停止不解除电机使能，也不能替代硬件急停。平台不包含碰撞规划或自动任务成功判定，
默认由操作者在任务完成后手动停止。

## 代码与验证

| 路径 | 用途 |
|---|---|
| `piper_studio/inference.py` | 后台推理与控制调度 |
| `piper_studio/inference_stream.py` | 时间对齐、PCHIP 与动作块衔接 |
| `piper_studio/policy_client.py` | 长连接与无损 RGB 传输 |
| `piper_studio/inference_app.py` | 预测、执行和运行回看 UI |
| `server/serve_openpi.py` | 可配置的 GPU 服务入口 |
| `tests/` | 插值、超时、通信与设备回归测试 |

```bash
python -m pip install -e '.[test,mac]'  # Linux 将 mac 换为 hardware
python -m pytest -q
```

另见[验证记录](docs/VALIDATION.md)与[第三方说明](THIRD_PARTY_NOTICES.md)。
UI 截图和本地回归测试不代表真机任务成功率。
