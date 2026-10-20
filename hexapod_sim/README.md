# 简化六足 MuJoCo 模型

打开模型（从 `RL` 目录运行）：

```bash
.venv-mujoco/bin/python -m hexapod_sim.fuzhu.view_model
```

`model.py` 根据同一套坐标参数在内存中生成 MuJoCo 模型，`fuzhu/view_model.py` 显示 HOME 静态站姿。
机身用圆柱、腿用胶囊体表示，不使用 STL。鼠标拖动旋转视角，滚轮缩放。
当前固定机身、不执行物理步进；质量和惯量尚未标定，不能作为动力学训练结果。

从 `RL` 目录运行：

```bash
.venv-mujoco/bin/python -m hexapod_sim.fuzhu.verify
```

`kinematics.py` 保存一套单腿 FK/IK，由六组 `LEGS` 安装数据复用。
长度使用米，几何关节角使用弧度；机身 +X 向前、+Y 向左、+Z 向上。
腿号 0～5 对应左前、左中、左后、右后、右中、右前，关节 ID 为 `3*i+[0,1,2]`。

变换链：

```text
T_body_foot = T(mount) Rz(mount_yaw) Rz(q_hip)
              Tx(0.042) Ry(-q_thigh)
              Tx(0.062) Ry(-q_knee) Tx(0.120)
```

单腿局部 +X 指向机身外侧，+Z 向上。三个几何角均为零时腿水平伸直。
髋关节绕局部 +Z 转动，大腿/小腿绕局部 -Y 转动；采用小腿向下弯的 IK 分支。
HOME 保留固件约定：相对髋原点，但轴方向与机身平行。先加 mount，得到机身坐标目标。

尺寸、安装位置、HOME、校准值来自当前 `6zu/hexapod-curve.ino`、
`6zu/src/config.h`、`6zu/src/kinematics/ik.cpp` 的 hexapod_type=1 配置。
`servo_degrees` 单独处理固件舵机角映射，不做静默限幅。
物理关节限位、质量、惯量、执行器响应尚待测量/配置。

## 第一步：重力下保持 HOME

```bash
# 查看30秒动态站立（可修改时长）
.venv-mujoco/bin/python -m hexapod_sim.fuzhu.stand --seconds 30
# 无窗口验证
.venv-mujoco/bin/python -m hexapod_sim.fuzhu.stand --headless --seconds 5
```

`make_model(dynamic=True)` 使用浮动机身、重力、圆柱/胶囊碰撞体、球形足端，
18个位置执行器保持几何 HOME。惯量由 MuJoCo 根据指定几何形状和质量计算。
每腿采用相同的估计质量，参数集中在 `physics.py`：

| 参数 | 暂定值 |
| --- | --- |
| 机身质量 | 1.2 kg |
| 每腿髋/大腿/小腿质量 | 0.08 / 0.10 / 0.06 kg |
| 每脚附加球体质量/半径 | 0.005 kg / 8 mm |
| 总质量 | 2.67 kg |
| 滑动摩擦系数 | 0.8 |
| 位置增益 kp / kv | 20 N·m/rad / 0.5 N·m·s/rad |
| 力矩上限 | ±1.5 N·m |
| 物理步长 | 2 ms |

上述物理数值是调试假设，不是实测电机/机械参数。
关节和目标范围由固件的0～180度指令窗口反算，不代表已验证的机械安全限位。
关节方向、几何参数和CAL映射不变。该步骤尚未模拟200ms下发、速度限制、
电机内部插补或串口延迟，尚未接入步态和强化学习。
相邻直接连接刚体由 MuJoCo 默认过滤碰撞，其他自碰撞保留。
`fuzhu/stand.py` 输出站立高度、倾斜、关节跟踪误差、接触数和数值警告，结束后自动关闭窗口。

## 第二步：Arduino 三足步态

```bash
.venv-mujoco/bin/python -m hexapod_sim.run --seconds 30
.venv-mujoco/bin/python -m hexapod_sim.run --headless --seconds 8
```

`gait.py` 移植当前 `6zu/src/locomotion/gaits.cpp` 的三足轨迹，`run.py` 负责运行：
腿0/2/4和腿1/3/5交替，周期800ms，每20ms规划一次，半周期20个tick，
XY余弦轨迹、摆动Z正弦抬腿，包含固件整数除法和 tick 未到端点即切相的行为。
先保持 HOME 1秒，再开始行走。默认内部命令X=40（BLE乘0.5之后的量），
对应28mm名义步幅、17mm步高。默认每200ms更新一次位置执行器目标。
`--x`/`--y`/`--r` 改内部命令；`--send-ms 20` 可对照逐规划帧下发。

沿用已有几何IK和Arduino舵机角映射，仿真中执行器接收几何关节弧度。
没有IMU补偿、没有RL；未模拟STM32运动时间插补、串口延迟和实机速度限制。
当前结果只能验证该估计物理模型中的步态，不能代表实机行走效果。

## 地形与键盘控制

```bash
# 组合场地，键盘控制，启动时站立不动；10分钟后自动结束
.venv-mujoco/bin/python -m hexapod_sim.run --terrain course --keyboard --seconds 600
# 全场视角；不加 --overview 时镜头跟随机器人
.venv-mujoco/bin/python -m hexapod_sim.run --terrain course --keyboard --overview --seconds 600
# 无窗口自动测试，内部X命令默认40
.venv-mujoco/bin/python -m hexapod_sim.run --terrain ramp --headless --seconds 85
```

启动后会出现MuJoCo观察窗口和独立的“六足机器人控制”窗口。
点击控制窗口后操作键盘，也可以点击方向按钮。MuJoCo窗口不再注册机器人键盘回调，
其原生工具栏和快捷键保留，不会同时触发机器人命令。
方向键采用锁存方式：按一次持续运动，松开不停止；切换到其他窗口会自动停止。
每次只选择一个运动方向，不叠加多键。命令相对机器人机身，不是固定世界方向。

| 按键 | 功能 |
| --- | --- |
| W / S | 前进 / 后退 |
| A / D | 左移 / 右移 |
| Q / E | 正 / 负旋转命令，沿用固件符号 |
| 空格 | 停止步态，目标回HOME，保留当前世界位置 |
| R | 重置回初始出生点、清零速度、站立1秒；属于仿真重置 |
| +（或=）/ - | 调大 / 调小内部命令强度，范围10～150，默认40，每次5 |
| Esc | 在控制窗口中退出仿真；关闭控制窗口也会退出 |

控制窗口的空格会让名义足端回HOME，不是仿真暂停；200ms下发时序仍生效。
窗口左上角显示地图、当前命令及机身离地高度。跌倒后停止更新步态，按R重新开始。
键盘模式忽略 `--x/--y/--r`，始终从停止开始。

强度上限150，仍沿用Arduino的127满量程：缩放后X/Y步幅限制±90mm，
转角参数限制±35°。所以127～150产生相同轨迹，不会继续提速。
纯前进步高为 `clip(10+abs(sx)/4,5,25)` mm，强度85起达到25mm上限；
振幅为 `sx/2`，纯前进最大45mm，未达到后续±50mm振幅限幅。
完整周期固定800ms、支撑相400ms，提高强度不提高迈腿频率。
无滑移且完整跟踪连续轨迹时，支撑相平均速度估计为步幅/0.4s；
实际速度受200ms目标采样、位置伺服、摩擦和打滑影响，需要测量。

地图 `--terrain` 可选：

| 地图 | 布局（世界+X方向） |
| --- | --- |
| flat | 平地 |
| obstacles | X=0.50m、0.80m放横向障碍，默认高度8mm、12mm，宽度0.70m |
| ramp | X=0.50m开始上坡，坡长0.60m、平台长0.35m，再下坡0.60m；默认5° |
| course | 先过两条障碍，再从X=1.15m开始上坡、平台、下坡 |
| steep | 难坡路线：10°、15°、20°上下坡，分别从X=0.50/1.75/3.00m开始 |
| stairs | 50mm一级的阶梯：三级上台阶、顶部平台，再三级下台阶回地面 |

阶梯从X=0.50m开始，每级升高50mm（约为HOME机身原点离地100mm的一半），
踏面长250mm，顶部平台长500mm，总高150mm，宽700mm，结束于X=2.00m。
参数在 `terrain.py` 的 `STAIR_HEIGHT`、`STAIR_TREAD`。
阶梯高度不是机身圆柱厚度的一半，也不是沿用25mm最大步高。

```bash
.venv-mujoco/bin/python -m hexapod_sim.run --terrain stairs --keyboard --overview --seconds 600
```

```bash
.venv-mujoco/bin/python -m hexapod_sim.run --terrain steep --keyboard --overview --seconds 600
```

难坡从黄到橙再到红依次加陡。每组上坡水平长0.35m、平台0.25m、下坡0.35m，
组间平地0.30m，宽度0.70m；对应高度约62/94/127mm。
`steep`采用固定10/15/20°梯度，不受 `--slope-deg` 影响。
机器人、摩擦、执行器与原场地相同，未加入专门的爬坡控制；需要实际测试是否卡住、偏航或跌倒。

`--slope-deg 8` 修改坡度（支持0～20°，不含0），`--obstacle-mm 10` 修改第一障碍高度，
第二障碍为其1.5倍。斜坡是有碰撞的凸楔形体，不使用外部STL。
所有地图出生点都保留平地；地面红/绿标记对应世界+X/+Y。
地图只是人工测试地形，控制器尚未加入地形感知或自适应。
非键盘模式跌倒或出现数值警告时返回失败；输出完成只代表运行结束，是否越过全场需检查位移。

## 目录职责

```text
hexapod_sim/
├── __init__.py
├── kinematics.py     # 共用坐标关系、FK/IK、CAL映射
├── physics.py        # 估计质量、摩擦、执行器参数
├── model.py          # 构建机身、六腿、关节和执行器
├── terrain.py        # 平地、斜坡、障碍物、组合路线
├── gait.py           # 三足轨迹，只计算目标关节角
├── keyboard.py       # 独立控制窗口、按键/按钮到运动命令
├── run.py            # 动态仿真主入口、时序、镜头和结果
├── fuzhu/            # 不参与正常动态运行的辅助工具
│   ├── __init__.py
│   ├── view_model.py # 静态HOME预览入口
│   ├── stand.py      # 动态站立检查
│   ├── verify.py     # 运动学检查
│   └── verify_sim.py # 地形碰撞、键盘状态、出生点站立检查
└── README.md
```

运行检查：`.venv-mujoco/bin/python -m hexapod_sim.fuzhu.verify_sim`。
强化学习实现已清理，后续从当前普通仿真基线重新设计。
# 固定装配误差（强化学习前的仿真准备）

控制侧 `kinematics.py` 的安装位置、连杆长度、HOME、CAL保持不变。
`perturbations.py` 启动时独立均匀采样三类误差，每轮固定，R重置不重采样：

- `--joint-bias-deg 2`：18个舵机角各自±2°，在IK之后、执行器目标限幅之前添加。
  按Arduino舵机角定义，换成MuJoCo几何角时髋取负号，大腿/小腿取正号。
- `--mount-bias-mm 2`：每条腿实际髋安装位置在机身XYZ各自±2 mm，不包含安装角误差。
- `--length-bias-mm 2`：每条腿三段实际连杆长度各自±2 mm；碰撞体及后续关节位置同步改变。
- `--seed 42`：复现同一组误差；换种子模拟另一台机器人。默认三类幅度均为0。

HOME是名义足端目标，不额外扰动；误差会自然造成实际HOME足端偏离。
动态初始化仅在必要时整体抬高，避免足球穿地，不消除相对误差；随后由重力落地。
质量和控制增益暂不随机化。此处是误差环境，不是已经加入强化学习补偿。

在 `RL` 目录由用户执行：

```bash
.venv-mujoco/bin/python -m hexapod_sim.fuzhu.verify_perturbations
.venv-mujoco/bin/python -m hexapod_sim.run --keyboard --seconds 600 --terrain flat --seed 42 --joint-bias-deg 2 --mount-bias-mm 2 --length-bias-mm 2
```

去掉三个bias参数即为无误差对照。可只保留一种，分别观察影响。
静态查看也支持同样的偏置参数：`-m hexapod_sim.fuzhu.view_model`。
