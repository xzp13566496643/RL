"""MuJoCo物理参数总表：常调参数集中在本文件，长度用m、质量用kg、角度用deg。"""

# 仿真与接触
TIMESTEP = 0.005                      # 200 Hz物理仿真
GRAVITY = (0.0, 0.0, -9.81)
CONTACT_SOLREF = '0.01 1'
CONTACT_SOLIMP = '0.9 0.95 0.001'
CONTACT_IMPRATIO = 10.0       # 摩擦约束相对法向接触约束的硬度
NOSLIP_ITERATIONS = 5         # 常规求解后用于抑制静态漂移的无滑移迭代次数
DEFAULT_JOINT_DAMPING = 0.02
DEFAULT_JOINT_ARMATURE = 0.0001
INITIAL_GROUND_CLEARANCE = 0.002

# 机身和三段腿
BODY_MASS = 1.2
BODY_RADIUS = 0.079
BODY_HALF_HEIGHT = 0.018
BODY_HOME_HEIGHT = 0.0916
LINK_MASSES = (0.08, 0.10, 0.06)       # 髋、大腿、小腿
LINK_RADII = (0.009, 0.008, 0.006)

# 独立站立平台：半尺寸对应完整尺寸1.6×1.6×0.04 m
PLATFORM_HALF_SIZE = (0.8, 0.8, 0.02)
PLATFORM_TOP_Z = 0.7

# 被动万向圆盘足端
FOOT_BALL_RADIUS = 0.008
FOOT_PAD_RADIUS = 0.030                 # 半径30 mm，直径60 mm
FOOT_PAD_HALF_THICKNESS = 0.002         # 总厚度4 mm
FOOT_PAD_MASS = 0.015
FOOT_PAD_MAX_TILT_DEG = 25.0
FOOT_PAD_DAMPING = 0.002

# 摩擦：MuJoCo顺序为“滑动、扭转、滚动”。主要防滑时调第一个数。
DEFAULT_FRICTION = '0.8 0.01 0.001'     # 普通机身和腿杆
PLATFORM_FRICTION = '1.5 0.01 0.001'    # 站立平台
FOOT_PAD_FRICTION = '1.5 0.01 0.001'    # 圆盘底面

# 18个主动关节的位置执行器（PD）
POSITION_KP = 20.0
POSITION_KV = 0.5
MAX_TORQUE_KGFCM = 20.0
MAX_TORQUE = MAX_TORQUE_KGFCM * 0.0980665  # 20 kgf·cm = 1.96133 N·m
