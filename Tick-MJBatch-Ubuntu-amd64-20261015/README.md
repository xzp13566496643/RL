# Tick 六足机器人 + mjbatch：Ubuntu amd64 部署使用说明

交付日期：2026-10-15。用途：研究用模型展示、CPU 批量仿真和短时强化学习试跑。

## 1. 交付状态

本包包含源代码、模型、已有短训练策略、截图和历史日志。**按委托要求，本次仅整理打包，没有重新安装依赖、启动程序、运行测试或在 Ubuntu 上做功能校验。**

此前在 macOS ARM64 上已运行：64 个并行环境、8 个 CPU 工作线程、30 次 PPO 更新，共 61,440 条交互样本。训练约 6 秒仅代表原机器当时结果，不代表 Ubuntu 性能。策略尚未收敛，暂时不能稳定行走。

本包提供本地编写的 mjbatch/PPO 适配脚本；不是上游 ARC-RL 完整 SAC 训练实验的复现。

## 2. 推荐系统

- Ubuntu **24.04 amd64 / x86_64**，推荐使用系统 Python 3.12。
- 安装脚本接受 Python 3.11、3.12、3.13。Ubuntu 22.04 默认 Python 3.10，需要另行准备上述版本之一，配套的 venv 和开发头文件也应匹配。
- 普通多核 CPU 即可，默认使用 CPU 版 PyTorch，不要求 NVIDIA GPU 或 CUDA。
- 查看图形界面需要可用的桌面、X11/Wayland 与 OpenGL。仅 SSH 的机器优先用无窗口训练，或另行配置 VNC/远程桌面。
- 需要联网下载 Python 依赖和构建依赖。这不是包含全部依赖的离线安装包。

## 3. 解压与安装

```bash
unzip Tick-MJBatch-Ubuntu-amd64-20261015.zip
cd Tick-MJBatch-Ubuntu-amd64-20261015

sudo apt-get update
sudo apt-get install -y build-essential pkg-config python3-dev python3-venv   libgl1 libglfw3 libegl1 libgles2 libosmesa6 mesa-utils

bash setup-ubuntu.sh
```

如需选择解释器，例如 Ubuntu 24.04 上的 Python 3.12：

```bash
PYTHON_BIN=python3.12 bash setup-ubuntu.sh
```

脚本在本包根目录创建 `.venv/`，安装 `MuJoCo 3.14.0`、`NumPy 2.4.6`、`Pillow 12.3.0`、`PyTorch 2.9.0 CPU`，再从随包的 `vendor/mjbatch` 编译安装 mjbatch 0.1.4。CMake、nanobind、scikit-build-core 等构建依赖由 pip 根据上游 pyproject.toml 下载。

不要复制 Mac 的虚拟环境或编译扩展到 Linux，本包已排除这些文件。

## 4. 先看模型和已有策略，不重新训练

在有桌面的终端中执行：

```bash
bash run.sh play --seconds 30
```

程序读取已提供的 `artifacts/tick-mjbatch/tick-short-ppo.pt`，显示 Tick 模型及已有短训练策略的动作。30 秒后自动关闭；也可以点击窗口关闭按钮。预览不会追加训练。

## 5. 做一次短训练

```bash
bash run.sh train --num-envs 64 --updates 30 --seconds 8
```

这会使用 64 个独立仿真环境做 30 次 PPO 更新，训练中显示其中 16 个状态，结束后再预览动作约 8 秒并自动关闭。显示时对各实例添加平移偏置，物理环境之间没有相互作用。

`--updates` 是 PPO 更新次数，每次每个环境采集 32 个控制步；`--seconds` 仅控制训练结束后的预览时长，**不是训练时间上限**。训练耗时取决于目标机器性能。

资源较少时可先运行：

```bash
bash run.sh train --num-envs 16 --updates 5 --seconds 5
```

重新训练会覆盖同名策略、截图和结果文件。若希望保留随包的初始结果，可先复制 `artifacts/tick-mjbatch/` 到其他目录。

## 6. 没有桌面的机器

```bash
bash run.sh headless --num-envs 64 --updates 30
```

该模式运行批量仿真和训练，不创建图形窗口或新截图。查看已有 PNG 不需要启动程序。

如工程师需要在无物理显示器的机器上生成渲染图，可自行配置 Xvfb：

```bash
sudo apt-get install -y xvfb xauth
LIBGL_ALWAYS_SOFTWARE=1 xvfb-run -s '-screen 0 1280x720x24'   bash run.sh train --num-envs 16 --updates 5 --seconds 5
```

这只是提供的部署方式，本次没有在目标系统验证。现有窗口由 GLFW 创建，使用 `--headless` 或 Xvfb；仅设置 `MUJOCO_GL=egl` 不会让 GLFW 桌面窗口自动变成无窗口渲染。

## 7. 工程师可选检查

以下命令留给部署者按需要运行，本次打包未执行：

```bash
bash run.sh check
```

用于比较单路 MuJoCo 与 mjbatch 状态并检查选择性重置。注意：现有 demo 在正常启动时也会执行一个简短的一致性检查；本次没有调用 demo，因此没有执行该检查。

## 8. 文件结构

```text
README.md                        本说明
setup-ubuntu.sh                  安装脚本
run.sh                           启动入口
requirements.txt                 本项目运行依赖（Torch 单独安装 CPU 版）
MANIFEST.sha256                  随包文件摘要
BUNDLE_INFO.json                来源、范围与打包状态
experiments/tick_mjbatch/demo.py  六足环境、短 PPO 与可视化
vendor/mjbatch/                  C++/Python 源码、构建配置、许可证、测试等
vendor/mjbatch/examples/window.py  窗口与渲染工具，demo 的直接依赖
vendor/ARC_RL/src/models/tick_v1.xml  未修改的原始 Tick 模型
vendor/ARC_RL/src/               上游模型、训练环境、算法与控制器代码供参考
artifacts/tick-mjbatch/          已有策略、五张截图、历史结果与日志
docs/MAC_TEST_RECORD.md          原 Mac 试跑记录，仅作历史资料
```

Tick 为浮动机身六足模型，共 18 个扭矩驱动、25 个 qpos 分量、24 个速度分量。物理步长为 0.002 秒，每个控制步运行 25 个物理步。

适配环境的 67 维观测包括姿态/关节状态、速度、上一动作和步态相位；奖励参考上游 Tick 的速度、生存、步态等项。此次适配没有包含上游的外力观测和接触力惩罚，步态接触使用足端高度启发式。后续正式训练可由工程师继续完善。

## 9. 输出与历史资料

- `01-model-loaded.png`：模型加载。
- `02-training-start.png`：第 1 次 PPO 更新。
- `03-training-middle.png`：第 15 次更新。
- `04-training-complete.png`：第 30 次更新。
- `05-policy-preview.png`：短训练策略动作。
- `tick-short-ppo.pt`：已有模型权重，可直接预览。
- `result.json`、`run.log`：原 Mac 试跑结果。
- `checks.json`、`validation.json`：原 Mac 检查结果。
- `requirements-freeze.txt`：原 Mac 环境记录，**不要直接用它安装 Ubuntu 环境**；请用本包 setup-ubuntu.sh。

历史 JSON 中可能保留原开发机平台、路径和版本信息，不是目标系统的验证报告。

## 10. 常见部署问题

- `glfw` / display 错误：在桌面会话运行，或改用 `headless` / Xvfb。
- OpenGL 驱动不可用：安装 Mesa；在有显示会话时可尝试 `LIBGL_ALWAYS_SOFTWARE=1 bash run.sh play --seconds 30`。
- C++ 或 Python.h 构建错误：确认 build-essential 和所选 Python 对应的开发头文件、venv 包已安装。
- MuJoCo 版本不匹配：保持 `mujoco==3.14.0`，在当前 `.venv` 中重新安装 `-e ./vendor/mjbatch`。
- PyTorch 下载慢：安装命令使用官方 CPU wheel 源；由部署者根据所在网络配置可用代理或镜像，不要误装完整 CUDA 依赖。
- 无执行权限：以上命令都通过 `bash script.sh` 调用，不依赖 ZIP 是否保留可执行位。

## 11. 上游与许可证

- mjbatch：https://github.com/kevinzakka/mjbatch ，随包代码来自 commit `088e2d4cd0d5878ea184010446c058710f23a671`，许可证见 `vendor/mjbatch/LICENSE`。
- ARC-RL：https://github.com/CarloRomeo427/ARC_RL ，随包代码来自 2026-10-15 下载的 main 源码包，许可证见 `vendor/ARC_RL/LICENSE`。
- PyTorch CPU 安装命令参考：https://pytorch.org/get-started/previous-versions/ 。

本包未包含上游演示视频、原机械臂项目的资源、开发机凭据和飞书发送记录。上游源码中的独立实验可能有额外依赖，本包安装脚本只安装 Tick/mjbatch 适配 demo 所需依赖。
