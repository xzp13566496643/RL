# ARC-RL Tick 六足机器人 / mjbatch 短训练试跑

已完成模型加载、批量物理一致性验证、64 环境短 PPO 训练及本机可视化。原始 Tick MJCF 未修改。

## 文件位置

- 上游源码：`vendor/ARC_RL/`。通过 GitHub 官方源码压缩包取得（git 传输连接失败），包含 LICENSE、模型与原训练代码。
- 模型：`vendor/ARC_RL/src/models/tick_v1.xml`。
- 独立适配脚本：`experiments/tick_mjbatch/demo.py`。
- 独立环境：`vendor/mjbatch/.venv/`，原机械臂环境未修改。
- 输出：`artifacts/tick-mjbatch/`，包含五张 PNG、检查结果、训练日志、策略权重与依赖清单。

## 再次查看界面

在 Roboarm 项目根目录执行：

```bash
vendor/mjbatch/.venv/bin/python experiments/tick_mjbatch/demo.py --play --seconds 30
```

重新打开已保存策略的可视窗口，30 秒后自动关闭，也可提前点击窗口关闭按钮。此命令不继续训练。

## 再次短训练

```bash
vendor/mjbatch/.venv/bin/python experiments/tick_mjbatch/demo.py --num-envs 64 --updates 30 --seconds 8
```

训练中显示 16 个环境的状态，实际参与训练的有 64 个独立环境。各机器人为显示加入平移偏置，物理世界之间没有相互作用。渲染来自真正的 mjbatch 状态，不是仓库演示视频。

## 检查与无窗口运行

```bash
vendor/mjbatch/.venv/bin/python experiments/tick_mjbatch/demo.py --check
vendor/mjbatch/.venv/bin/python experiments/tick_mjbatch/demo.py --headless --num-envs 64 --updates 30
```

图形窗口需要桌面和 OpenGL；本次在 macOS ARM64 验证，并未在 Ubuntu 重跑 Tick。

## 本次实测

- MuJoCo 3.14.0、mjbatch 0.1.4、PyTorch 2.9.0、Python 3.11.9。
- 六条腿、18 个扭矩驱动；`nq=25`、`nv=24`。保留原始 0.002 秒步长及求解器配置。
- 单路 MuJoCo 对比 mjbatch：40 个控制周期、每周期 25 个物理步，qpos 最大误差为 0。选择性重置验证通过，无非有限状态。
- 64 个环境，8 个 CPU 工作线程，30 次 PPO 更新，每次 32 个控制步：共 61,440 条交互样本、1,536,000 个物理步。
- 训练阶段约 6.12 秒，包含训练过程中的渲染和截图，不含之后的动作预览；不作为通用性能基准。
- 检查点重载通过，并确认 actor 权重发生改变；最后记录的 MuJoCo 警告数为 0。
- 两次图形试跑均正常结束并释放窗口、渲染和 GLFW 资源；未残留 demo 进程。

## 训练范围

这是为接入 mjbatch 添加的小型 PPO 验证程序，并非复现上游完整 SAC 实验。观测为 67 维，包含姿态/关节状态、速度、上一动作和步态相位；奖励参考上游 Tick 的速度、生存和步态项，未包含上游外力观测/接触力惩罚。步态接触使用足端高度启发式，和上游代码一致。

仅做了很短的训练，未获得收敛步态。平均训练奖励没有显示稳定提升，画面中会出现探索姿态和失稳；不能据此称为已经学会行走。

## 截图

- `01-model-loaded.png`：模型加载。
- `02-training-start.png`：第 1 次更新后的批量状态。
- `03-training-middle.png`：第 15 次更新。
- `04-training-complete.png`：第 30 次更新。
- `05-policy-preview.png`：短训练策略动作预览。

PNG 从实际 MuJoCo 窗口的渲染帧缓冲读取，未使用系统桌面截屏权限。

上游：https://github.com/CarloRomeo427/ARC_RL
