# 六足机器人 V3 Stage 2a（MJX/PPO）

这是依据服务器最后一次 Stage 2a 训练快照重建的精简工程。目录保留模型构建、MJX 环境、PPO、奖励、训练入口、必要配置，以及用于成果展示的浏览器实时仿真；不包含诊断和临时测试脚本。

## 目录

```text
configs/
  flat_walk_v3.yaml       # 任务、奖励和速度命令
  mjx_flat_walk_v3.yaml   # PPO及并行训练参数
hexapod/
  mjx/                    # 环境、网络、PPO、奖励、训练入口
  demo/                   # 浏览器实时仿真和Actor推理
  simulation/             # MuJoCo模型、运动学与物理参数
  rl/metrics.py           # 可选的训练曲线生成
artifacts/
  stage2a_actor_10000.npz # 已导出的第10000轮演示Actor
checkpoints/stage2a-10000/
  latest.pkl              # 服务器第10000轮完整训练状态
  latest-policy.npz       # 第10000轮策略权重
requirements.txt
```

## 浏览器成果演示

演示使用真实 MuJoCo 物理和第10000轮确定性 Actor，支持前进、后退、左移、右移和静止。速度滑块范围与 Stage 2a 训练范围一致，为 0.1–0.2 m/s。

服务器启动：

```bash
cd /home/xuzhanpeng/Hexapod_robot/hexapod_mjx_stage2a

python3 -m venv .venv-demo
.venv-demo/bin/python -m pip install -U pip
.venv-demo/bin/python -m pip install -r requirements-demo.txt

MUJOCO_GL=egl \
.venv-demo/bin/python -m hexapod.demo.server \
  --host 0.0.0.0 \
  --port 8000
```

随后在浏览器打开：

```text
http://服务器IP:8000
```

页面操作：

- 页面方向按钮：前进 `+X`、后退 `-X`、左移 `+Y`、右移 `-Y`。
- 静止按钮向策略发送 `(0, 0, 0)`。
- 键盘支持 `W/A/S/D`，松开后自动静止；空格立即静止。
- 页面实时显示目标/实际 XY 速度、速度大小、机身姿态、接地足数和仿真时间。

如果服务器端口不对外开放，可从本机建立 SSH 隧道：

```bash
ssh -L 8000:127.0.0.1:8000 xuzhanpeng@192.168.2.100
```

此时浏览器访问 `http://127.0.0.1:8000`。

演示权重已经导出。如果以后替换 checkpoint，可重新导出：

```bash
.venv-mjx/bin/python -m hexapod.demo.export_policy \
  checkpoints/stage2a-10000/latest-policy.npz \
  artifacts/stage2a_actor_10000.npz
```

## 环境安装

建议使用 Python 3.10 和 CUDA 12：

```bash
cd /home/xuzhanpeng/Hexapod_robot/hexapod_mjx_stage2a
python3.10 -m venv .venv-mjx
.venv-mjx/bin/pip install -U pip
.venv-mjx/bin/pip install -r requirements.txt
```

## 从头训练 Stage 2a

`CUDA_VISIBLE_DEVICES=1` 后，JAX 输出的 `CudaDevice(id=0)` 就是物理 GPU 1，这是正常的设备重映射。

```bash
cd /home/xuzhanpeng/Hexapod_robot/hexapod_mjx_stage2a
CUDA_VISIBLE_DEVICES=1 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
.venv-mjx/bin/python -m hexapod.mjx.train \
  --config configs/mjx_flat_walk_v3.yaml
```

## 从第10000轮继续训练到第15000轮

`--iterations` 表示最终总轮数，不是额外轮数：

```bash
cd /home/xuzhanpeng/Hexapod_robot/hexapod_mjx_stage2a
CUDA_VISIBLE_DEVICES=1 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
.venv-mjx/bin/python -m hexapod.mjx.train \
  --config configs/mjx_flat_walk_v3.yaml \
  --resume checkpoints/stage2a-10000/latest.pkl \
  --iterations 15000
```

恢复训练会继续写入 checkpoint 所在目录。若要保留随提交附带的原始 checkpoint，请先把该目录复制为新的运行目录，再对副本执行 `--resume`。

## 本版本的关键训练设置

- 256 个并行环境，rollout 25 帧，每轮 6400 个样本。
- 控制周期 0.02 s，物理步长 0.005 s，每个控制帧 4 个物理子步。
- Stage 2a 命令：50% 静止、50% 轴向 XY，速度 0.1–0.2 m/s。
- Tripod 步态 1.25 Hz，A 组腿 0/2/4，B 组腿 1/3/5。
- Actor 观测 65 维，Critic 观测 94 维，动作 18 维。
- Phase Positive 1.2、Action Rate 0.3、Action Magnitude 0.08。
- Foot Height 使用目标 0.03 m 处为峰值的正向二次奖励。
- Linear Velocity 保留原始公式，没有六足接地门控。

正式运行前可检查 GPU：

```bash
nvidia-smi
```
