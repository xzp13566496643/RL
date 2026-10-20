# 六足 MuJoCo 仿真

从本目录启动（当前环境已安装在 `.venv-mujoco`）：

```bash
.venv-mujoco/bin/python -m hexapod_sim.run --terrain course --keyboard --seconds 600
```

`hexapod_sim/README.md` 包含完整参数、地图布局、键盘操作和文件职责。
当前仅运行传统三足步态，保留平地、坡道、障碍物和阶梯场景。
本次新增的强化学习实现、训练环境及结果已清理，后续从头设计。

## 完整调用流程和代码定位

```text
run.main() 解析命令行参数
  ├─ model.make_model(dynamic=True)
  │    ├─ physics.py：质量、摩擦、PD位置执行器等估计参数
  │    ├─ terrain.add_terrain()：地面/障碍/斜坡碰撞几何
  │    └─ kinematics.py：六腿安装、连杆尺寸、HOME → 18关节初始角
  ├─ keyboard.Keyboard()：维护 X/Y/R、重置、退出请求
  ├─ mujoco.viewer.launch_passive()：显示，不注册机器人键盘回调
  ├─ keyboard.ControlWindow()：独立窗口接收机器人按键/按钮
  └─ 物理循环，每步2ms
       ├─ 每20ms：keyboard.poll() → X/Y/R
       ├─ 初始或重置后：HOME站立1秒
       ├─ 每20ms：gait.Tripod.step(X,Y,R)
       │    ├─ 三足相位、步幅、抬腿 → 六腿目标足端XYZ
       │    └─ kinematics.inverse() → 18个几何关节目标，单位rad
       ├─ 每200ms：关节目标限幅 → data.ctrl
       ├─ mujoco.mj_step()：位置伺服力矩、重力、接触、运动积分
       ├─ ground_height()和机身朝向：离地高度/跌倒检查
       ├─ 每20ms：更新镜头、状态文字、viewer.sync()
       └─ 结束：输出JSON结果、关闭窗口
```

| 文件/函数 | 负责内容 |
| --- | --- |
| `hexapod_sim/run.py:main` | 参数、主循环、时序、重置、退出、窗口显示 |
| `hexapod_sim/run.py:ground_height` | 向下射线测量当地地形高度 |
| `hexapod_sim/model.py:make_model` | 生成MJCF并加载为MuJoCo模型，无需外部模型文件 |
| `hexapod_sim/terrain.py:add_terrain` | 四种场地，障碍高度、坡角和碰撞体 |
| `hexapod_sim/keyboard.py:ControlWindow` | 独立控制窗口，按键/按钮事件及失焦自动停止 |
| `hexapod_sim/keyboard.py:Keyboard.on_key` | 接收独立窗口转发的命令请求 |
| `hexapod_sim/keyboard.py:Keyboard.poll` | 主循环读取键盘状态 |
| `hexapod_sim/gait.py:Tripod.step` | Arduino三足轨迹、相位更新、调用IK |
| `hexapod_sim/kinematics.py:inverse` | 机身坐标足端目标 → 几何关节角 |
| `hexapod_sim/kinematics.py:frames` | 几何关节角 → 各连杆/足端坐标，用于验证 |
| `hexapod_sim/kinematics.py:servo_degrees` | 映射Arduino舵机角，用于对照验证；当前动态仿真不走此函数 |
| `hexapod_sim/physics.py` | 估计物理参数，尚未实测标定 |
| `hexapod_sim/fuzhu/view_model.py` | 静态预览与图片导出 |
| `hexapod_sim/fuzhu/stand.py` | 动态HOME站立测试 |
| `hexapod_sim/fuzhu/verify.py` | 六腿FK/IK验证 |
| `hexapod_sim/fuzhu/verify_sim.py` | 地形碰撞、键盘状态及站立回归检查 |

机器人按键在独立控制窗口处理，也可以点击该窗口的方向按钮。
MuJoCo窗口不注册机器人键盘回调，原生快捷键只作用于查看器，避免同时触发机器人命令。
切换到其他窗口自动停止行走；关闭控制窗口或在其中按Esc退出仿真。
`viewer.sync()`也会同步面板设置；机器人目标角由我们的循环定期写入。

## 与实机的边界

几何、HOME、CAL和三足轨迹参考 `../6zu`。当前跳过BLE、UART、STM32和CAN链路，
不含IMU补偿、运动时间T插补及电机响应标定；位置执行器PD是仿真近似。
不是在运行Arduino二进制，也没有复现STM32内部控制。

## 清理范围

旧Tick/mjbatch强化学习演示包和使用说明已恢复到本目录，保留训练代码、策略权重和历史结果，
用于后续强化学习参考；当前简化模型不依赖该包。
`.venv-mujoco`为当前运行环境，必须保留。
`../6zu`为实机对照源码，`../meshes`为老师提供的原始素材，均未删除；
当前简化模型不读取这些STL。
