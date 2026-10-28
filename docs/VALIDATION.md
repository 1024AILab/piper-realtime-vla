# Validation / 验证记录

Date: 2026-10-28. Environment: Apple Silicon macOS, Python 3.12.14.

- Existing repository regression suite: **64 passed**.
- Command: `python -m pytest -q`, run from this repository using the existing
  local Python environment with the archived dependencies.
- Both application UIs were started locally in explicit simulation mode.
- No physical robot motion was triggered by these checks.
- GPU checkpoint loading and new hardware task-success rates were not measured
  during this backup. Model weights and training datasets are external inputs.

本次记录是拆分仓库后的本地回归与模拟界面检查。真机数据、模型权重和运行日志
未纳入公开仓库；截图不能替代真机成功率或模型推理性能测量。
