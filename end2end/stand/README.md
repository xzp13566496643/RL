# 端到端站立：第1步，动作执行环境

## Actor（已建立，尚未接Critic/PPO训练循环）

`actor.py`定义Tick风格的`StandActor`：81维观测经过两个64单元Tanh隐藏层，输出18维动作均值。
训练时另有18个可学习log_std构成独立高斯策略；部署时`deterministic()`使用均值并裁剪到[-1,1]。
输入顺序固定为60维归一化六轴IMU历史、18维上一动作、3维目标速度(vx,vy,wz)。
输出由env.py分段映射到每个关节的完整安全范围，零动作仍对应HOME。
输出层以小权重和零偏置初始化，使未训练策略的均值接近HOME；探索噪声只在训练采样时存在。

训练依赖尚未安装到`.venv-mujoco`；由用户执行：

```bash
.venv-mujoco/bin/python -m pip install -r end2end/stand/requirements-train.txt
.venv-mujoco/bin/python -m end2end.stand.test_actor
```

## Gymnasium接口（已封装，尚未接PPO）

`gym_env.py`的StandGymEnv继承gymnasium.Env，复用StandTask与StandEnv，不改变奖励或物理。
观测空间为Box(-100,100,(81,),float32)：60维归一化六轴IMU历史、18维上一动作和3维目标速度。
动作空间Box(-1,1,(18,),float32)，输入必须有限且在范围内，不符合则报错。
动作映射、限速、关节限位和局内零位误差仍由env.py处理。
没有固定HOME覆盖策略输入；action=0时才主动请求参考角。

接口：

```python
from end2end.stand.gym_env import StandGymEnv

env = StandGymEnv()
obs, info = env.reset(seed=42)
action = env.action_space.sample()  # 仅接口示例，不是学习策略。
obs, reward, terminated, truncated, info = env.step(action)
if terminated or truncated:
    obs, info = env.reset()
env.close()
```

首次reset(seed=42)固定Gym随机数序列，后续reset()派生新回合种子，
重复相同seed可复现整个序列。info.episode_seed可用于直接重现底层场景；
Gym seed通过派生后用于场景，因此不是同数字demo seed对应的场景。
terminated表示失败，truncated表示时长到期，不自动重置。
info保留真实状态及奖励分量，仅调试/评分使用，不能拼入策略观测。
目前不支持渲染模式，也不引入Torch或PPO依赖。

训练采用非对称Actor-Critic：Gym对Actor仍只返回81维实机可构造观测；info中的
`critic_observation`为149维训练专用观测。它由81维Actor观测加上局部地面夹角、机身
法向净空、6维足端接触、18维真实关节角、18维真实关节速度、18维PD实际目标和
6维各腿非足端碰撞组成。Critic网络为149→256→128→64→1，仅用于仿真训练，不能部署到ESP32。
68维特权部分采用固定尺度：夹角除以30°；净空以任务HOME目标为零点并除以20 mm容差；
真实关节角和PD目标按各关节完整安全范围映射；关节速度除以10 rad/s；接触保持0/1；
最后裁剪到±10。三个通用尺度集中在config.py的EnvironmentConfig，净空目标和尺度复用TaskConfig。

`rollout.py`提供固定长度、多环境的PPO轨迹缓存，保存Actor/Critic观测、动作、旧log概率、
旧价值、下一状态价值、奖励、terminated和truncated，并计算GAE与价值回报。真实失败不对
下一状态价值bootstrap；时间上限会bootstrap，但两种结束都会切断GAE，避免跨reset串联两局。
`collector.py`先实现单环境真实轨迹采集：调用Actor采样、Critic估值、Gym step和自动reset，
填满buffer后计算GAE。它只采集数据，不更新网络；多环境采集在单环境PPO闭环验证后再扩展。
`ppo.py`实现裁剪策略目标、裁剪价值损失、高斯基分布熵、随机mini-batch、多轮更新、
全局梯度裁剪和目标KL提前停止。一个Adam优化器管理两套独立网络，更新后将Actor的
`log_std`限制在[-3,0]；统计量会返回损失、熵、近似KL、裁剪比例和更新前梯度范数。

`train.py`串联单环境采集、GAE和PPO更新。每轮向`metrics.jsonl`追加一条JSON指标；
按配置周期原子保存`latest.pt`及带迭代号的快照，训练结束也保存latest。checkpoint包含
Actor、Critic、Adam、迭代/样本计数、完整配置及Python/NumPy/Torch随机状态。恢复发生在
rollout边界，会恢复学习状态并从新环境回合继续，不保存MuJoCo运行到一半的物理状态。

先做20轮流程试跑：

```bash
.venv-mujoco/bin/python -m end2end.stand.train --iterations 20 --rollout-steps 128 --mini-batch-size 64
```

正式按config.py默认值训练：

```bash
.venv-mujoco/bin/python -m end2end.stand.train
```

从latest继续（总迭代数必须大于等于checkpoint当前轮数）：

```bash
.venv-mujoco/bin/python -m end2end.stand.train --resume end2end/stand/checkpoints/run-时间/latest.pt
```

## 训练参数总表

训练侧需要反复调整的默认值集中在`config.py`：NetworkConfig控制Actor/Critic层宽、
激活函数和初始探索标准差；EnvironmentConfig控制周期、目标限速、三类误差和速度命令；
RolloutConfig控制步数、环境数、gamma、GAE lambda和优势归一化；PPOConfig控制学习率、
裁剪、损失系数、训练轮数、mini-batch、梯度和KL；TrainingConfig控制种子、设备、总迭代、
日志及checkpoint，并组合IMUConfig、SceneConfig和TaskConfig。TaskConfig内的RewardConfig
集中列出所有奖励/惩罚权重及尺度。机器人质量、摩擦、执行器和接触等模型真值仍只在
`hexapod_sim/physics.py`维护，不在训练配置复制第二份。

由用户在RL目录执行安装和检查：

```bash
.venv-mujoco/bin/python -m pip install -r end2end/stand/requirements-gym.txt
.venv-mujoco/bin/python -m end2end.stand.test_gym_env
```

官方check_env可能提示观测空间边界为无穷，这是保留原始IMU读数的设计，不是测试失败。
检查包括官方接口检查、种子序列、动作接管、观测空间及失败/超时区分。
API依据：https://gymnasium.farama.org/introduction/create_custom_env/

## 当前演示入口行为

默认demo立即推进物理，不冻结，保持本局随机初始关节目标作为无策略对照。
它不是无力矩自由落体：PD仍跟踪初始动作，但不再默认给HOME目标。
默认运行20秒，可用--seconds延长；--headless同样运行物理。
未来训练入口由策略动作驱动env.step；偏置仍在策略目标与PD执行器之间施加。
如需原固定角对照，显式加`--baseline-hold`；`--joint N`仍显式运行人工单关节测试。
check_task保留为明确命名的固定角评分基线，不是正式训练入口。

```bash
.venv-mujoco/bin/python -m end2end.stand.demo --seed 42 --seconds 60
.venv-mujoco/bin/python -m end2end.stand.demo --seed 42 --baseline-hold
```

所有演示模式都逐秒打印IMU；没有策略时不宣称主动平衡。

## 被动万向圆盘脚垫

动态模型的六条腿各增加一个无执行器ball joint，等效真实的低摩擦球窝结构。
球头半径8 mm，仅作连接外观且不参与碰撞；球心下方连接半径30 mm、厚4 mm的圆盘碰撞体。
圆盘半径30 mm、质量15 g，最大摆角25°，球关节阻尼0.002；圆盘与地形的滑动摩擦系数当前均为1.5。
圆盘由重力和接触力被动调姿，不读IMU、不受策略直接控制，策略动作仍严格为原18个关节。
圆盘零位按名义HOME预对准世界水平，斜坡接触后可绕球心自动倾转。
MuJoCo中没有直接模拟凹球壳与内球的接触，而使用理想球关节等效，以避免凹网格凸包化和内部接触不稳定。

加入6个球关节后，MuJoCo内部qpos不再是“自由基座后连续18个数”。
model.py中的controlled_qpos/set_controlled_qpos按关节名称定位18个主动关节，训练观测仍不包含任何关节位置。
仿真结构应与计划中的真机脚垫一致；如果真机不安装该结构，训练结果不能直接视为可迁移。

参数集中在hexapod_sim/physics.py：FOOT_PAD_RADIUS、FOOT_PAD_HALF_THICKNESS、
FOOT_PAD_MASS、FOOT_PAD_MAX_TILT_DEG和FOOT_PAD_DAMPING。

## 安全关节范围

地形与足端的滑动摩擦系数均暂设为1.5（hexapod_sim/physics.py中的
PLATFORM_FRICTION与FOOT_PAD_FRICTION）。CONTACT_IMPRATIO=10提高摩擦约束硬度，
NOSLIP_ITERATIONS=5修正接触后的残余漂移；均非实机标定值，后续sim-to-real前需要验证。

统一配置在 `RL/hexapod_sim/joint_limits.py` 的 `SERVO_LIMITS_DEG`：
髋60～120°，大腿30～120°，小腿30～150°。
这是以最终舵机90°（P1500）为零位换算的命令范围，已包含CAL，不是相对HOME。
模型按每腿CAL转换为几何角，同时用于关节range和执行器ctrlrange。
站立环境直接使用完整安全范围，并为本局固定零位偏置预留空间；名义命令和偏置后目标均受限。
动作采用以HOME为零点的分段映射：action=-1/+1到达对应几何方向的安全边界，action=0保持HOME。
更改该公共配置也会影响原三足步态仿真；越界步态目标会被原有执行器命令限幅裁剪。
这是仿真中的安全范围近似，不代表真实机械硬挡，也不能替代实机端安全保护。
MuJoCo关节限位采用约束求解，实际关节位置可能存在微小越界；命令限幅不等同于绝对机械保证。
`test_env`新增检查角度换算、模型限位和未裁剪HOME，以及偏置前后舵机目标边界。

本目录复用hexapod_sim的模型与误差配置，不修改原步态入口。
目前已有模拟IMU观测、任务奖励、回合管理及Actor/Critic网络，尚未接入PPO训练循环。

- env.py：reset生成每局固定误差，step接收18维动作并推进物理。
- demo.py：固定站姿/单关节正弦测试，可视化或无窗口运行。
- test_env.py：18个关节双向动作、限幅、限速、误差采样的自动检查。

## 动作定义

action形状(18,)，先裁剪到[-1,1]。
负、正半轴分别把HOME分段映射到该关节安全下限、上限，不是对上一帧累加。
之后限制目标变化速度（默认60°/s），最后施加本局舵机零位偏置。
Actor训练时从18维高斯分布采样原始动作，再经tanh平滑压缩到(-1,1)；PPO使用带
变量变换修正的log概率。部署时执行`tanh(均值)`，不使用随机标准差。
默认20 ms更新动作，每次4个5 ms物理步；期间按100 Hz产生2次IMU采样。
这是仿真暂定周期，不代表实机已验证。
限速只限制发送目标，不保证实际关节角速度或真实硬件安全。
角度采用MuJoCo几何角；接入实机时须转换Arduino舵机角，不能直接发送这些数值。

关节顺序：左前、左中、左后、右后、右中、右前；每腿髋/大腿/小腿。
例如0/1/2是左前腿，3/4/5是左中腿。
参考角只在环境初始化用名义IK计算一次，step完全不调用IK和gait。
state暴露真实状态用于调试，不是未来的策略观测，尤其不能将实际关节角作为实机可用输入。

## 用户执行

在RL目录运行，不需要安装新依赖：

```bash
cd /home/xuzhanpeng/Hexapod_robot/RL
.venv-mujoco/bin/python -m end2end.stand.test_env
.venv-mujoco/bin/python -m end2end.stand.demo
```

单关节测试（左前大腿，幅度为15°的30%，正弦周期4秒）：

```bash
.venv-mujoco/bin/python -m end2end.stand.demo --joint 1 --amplitude .3
```

固定站姿加入装配偏置：

```bash
.venv-mujoco/bin/python -m end2end.stand.demo --seed 42 --joint-bias-deg 2 --mount-bias-mm 2 --length-bias-mm 2
```

换seed再reset生成另一组误差；相同seed可复现。端到端站立默认关节偏置±3°，安装位置和长度偏置默认0。
`--joint-bias-deg 0`关闭关节偏置；这里的默认值不改变原hexapod_sim步态入口的默认值。
可用--headless、--seconds、--control-ms、--max-speed调整测试；动作范围统一来自安全关节限位。
演示仅在机身实际接触地形时记录首次失败，但继续物理运行，不关闭窗口或重置。
到--seconds时暂停在最后状态，保留窗口供鼠标观察，关闭窗口/Ctrl+C退出。
--headless在指定时长结束后正常退出；训练任务的失败/超时规则不变。
这些是仿真动作检查，不对真机下发任何命令。

## 100 Hz模拟IMU

`imu.py`的`IMUConfig`配置采样周期、历史长度、噪声标准差和零偏幅度。
机身原点的原生accelerometer/gyro模拟比力和角速度，轴向为+X前、+Y左、+Z上。
默认每10 ms采样，原始`imu.observation()`返回最近10帧、从旧到新展平的60维float32数组。
每帧顺序为ax/ay/az（m/s²）、gx/gy/gz（rad/s）。策略接口`env.observation()`返回81维：
加速度乘1/9.81、角速度乘0.25后的60维IMU历史，随后是18维上一动作和3维
目标速度(vx,vy,wz)；速度命令分别乘(2,2,0.25)。完整观测最终裁剪到±100，仅作异常保护。
站立任务的目标速度固定为(0,0,0)，不含高度命令。该方式参考Eigenbot的固定尺度预处理，
不使用运行均值/方差；与其不同的是，本项目的IMU噪声与每局零偏会真实加入观测。
启动历史不足时复制第一帧补齐；初始支撑尚未建立时可能处于自由落体，加速度不一定为g。
水平静止且受到支撑时az应接近+9.81，自由落体时比力接近0。
未实现姿态融合、滤波、温漂或实机安装轴变换；接实机须对齐单位、轴向和预处理。

默认加速度噪声标准差0.05 m/s²、角速度噪声标准差0.005 rad/s，逐采样变化。
零偏每轴分别在±0.1 m/s²、±0.01 rad/s采样，局内固定。
以上仅为起始仿真参数，必须以后用实机日志校准，不是设备噪声规格。
`reset()`不指定seed会换一组装配误差和IMU零偏，指定相同seed则复现同一组及噪声序列。
调用observation只读缓存，不重新生成噪声。噪声和零偏均可单独设置为0。

```bash
.venv-mujoco/bin/python -m end2end.stand.test_imu
.venv-mujoco/bin/python -m end2end.stand.demo --ideal-imu
.venv-mujoco/bin/python -m end2end.stand.demo
```

demo每秒打印一次六轴读数；最终输出零偏、采样总数和观测维度。
传感器语义参考MuJoCo官方文档：https://mujoco.readthedocs.io/en/stable/XMLreference.html#sensor-accelerometer

## 站立任务与回合

`task.py`中的`StandTask`包装物理环境，不修改demo或原步态。暂不依赖Gymnasium。
`reset(seed=None)`返回(81维策略观测, info)，不带种子则每局重新随机误差。
`step(action)`返回(observation, reward, terminated, truncated, info)。
terminated是失败，truncated是到时；结束后必须显式reset，不自动跨局。
TaskConfig默认10秒一局；HOME机身原点高97.2 mm、半厚18 mm，因此目标机身下表面净空为79.2 mm。
净空在目标±20 mm内均为满分，超出后按离边界的距离对称衰减。姿态或净空本身不提前终止，机身实际触地才失败。
重置从接近HOME姿态开始，不预先用控制器消除装配误差，不包含倒地起身。

奖励先按每秒速率计算：机身平行于正下方局部地面最多+2，净空匹配最多+1，
六足全部接地最多+0.5；另外惩罚机身线/角加速度、发送目标变化速度和腿杆等非足端地形碰撞。
乘控制周期得到每步奖励，失败额外扣5。各项系数在reward_terms中独立设置。
目标变化惩罚使用限速后的实际发送命令，不是关节真实速度。
高频运动项使用相邻控制周期的机身线速度和角速度变化率；足端接触按六只脚去重计数。
评分使用理想状态，与带噪IMU观测分离；info中的真实状态仅用于调试，不供策略输入。
当前在动作边界检查失败/接触，尚未实现所有物理子步内事件捕获。
数值异常则失败并返回上一帧有限观测，避免异常数据进入策略。
奖励系数是第一版设计，还需通过基线和训练检验，不能把高奖励当成实机可靠保证。

用户执行：

```bash
.venv-mujoco/bin/python -m end2end.stand.test_task
.venv-mujoco/bin/python -m end2end.stand.check_task
.venv-mujoco/bin/python -m end2end.stand.check_task --seed 42 --joint-bias-deg 2 --mount-bias-mm 2 --length-bias-mm 2
```

check_task无窗口运行3局固定角基线，依次使用seed、seed+1、seed+2，打印回报、
最大倾角、最低高度和结束原因。`--motion sine`可对比单关节正弦动作。
这不是PPO训练入口，下一步才加Gym适配与训练。

## 固定地形图集与随机初态（当前任务）

模型一次性建立4类×4级固定图集：平面、四向金字塔坡、随机砖面和离散矮障碍。
每块2×2 m，间距2.4 m，基准面与世界地面同为Z=0；坡度等级为3/7/11/15°，
起伏幅度最高15 mm，障碍高度最高20 mm。每局不重建地形，而是随机选择块、等级、
块内安全出生位置和机身Yaw；同一seed完全复现，适合以后多个环境共享同一MjModel。
默认30%抽平地，其余在坡面、起伏和障碍间采样；`--max-terrain-level`限制最高难度。
每局18个舵机零位偏置独立采样±3°，初始化和后续输出复用同一组，局内不变、不累加。
初始实际角=名义初始命令角+本局零位偏置；执行器目标=本次名义命令角+同一偏置。
独立初始关节扰动默认为每关节±20°：初始实际角=HOME+零位误差+大扰动（按限位裁剪）。
初始PD目标与实际关节角一致；后续策略给什么目标就执行什么，不再次叠加初始大扰动。
不额外旋转整台机器人：初始化机身保持水平但Yaw随机，随后姿态由关节、重力和接触决定。
扰动后按限位裁剪，并整体抬升/降低到所有碰撞体与地面最小法向间隙2 mm，避免穿地。
初始化因此可能短暂只有部分腿接地或略微悬空，随后由物理接触建立支撑，不做隐藏稳定预处理。
名义参考仍为HOME，零位偏置作为永久局内误差存在；±20°关节扰动只设置初始动作。
环境默认单独构造StandEnv时保持旧平地行为；StandTask默认及两个演示入口显式启用新场景。

评分姿态目标是机身Z轴平行于正下方射线命中的局部地面法向。
净空按该法向计算到圆柱机身外表面的近似最近距离，而不是世界Z方向的机身原点高度。
离开所选2×2 m地形块不再终止，块外为连续世界平面；state.on_platform仅供诊断。
state.height_m和tilt_deg保留世界坐标调试量，另提供ground_height_m、clearance_m、
body_clearance_m、ground_alignment_deg、ground_normal和slope_deg。真实地形量只进评分/info，不进策略观测。
当前没有倒地翻身、外力推压或已训练策略。

```bash
.venv-mujoco/bin/python -m end2end.stand.test_scene
.venv-mujoco/bin/python -m end2end.stand.demo --seed 42
.venv-mujoco/bin/python -m end2end.stand.check_task --seed 42 --joint-bias-deg 2 --mount-bias-mm 2 --length-bias-mm 2
```

--flat固定使用最低级水平地形块并关闭初始大关节扰动；不关闭IMU或关节零位误差。
完全无关节/机身扰动的平地对照使用`--flat --joint-bias-deg 0`，关闭IMU误差另加`--ideal-imu`（demo）。
--max-slope-deg、--joint-bias-deg、--initial-joint-deg可分别调整范围；
--flat-probability设置平地概率（0表示只从三类非平地抽样，1表示每局平地）。
--initial-body-deg和--body-tilt-probability已移除，图集最大坡度暂限制15°。
每个任务结束后重新reset将重采样；显式相同seed可复现。

当前训练任务确定为：初始关节大扰动+随机图集地形，学习恢复并维持相对当地地面平行的站立。
不含外力推压或倒地起身，不额外随机机身姿态。
坡面为四向3～15°，默认30%平地；地图固定，只随机每局所在块、等级、落点和Yaw。
评分参考机身下表面法向净空79.2 mm，满分带为59.2～99.2 mm，不要求恢复HOME关节角。
该任务是否能明显优于固定角PD基线须后续评估，当前没有开始PPO训练。
