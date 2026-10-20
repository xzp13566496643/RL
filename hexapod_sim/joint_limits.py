"""安全限位：以最终舵机命令角定义，按各腿CAL换算成MuJoCo几何角。

90度对应P1500，不是HOME；以下是用户提供的安全范围，不是实测机械硬挡。
"""
import numpy as np

from .kinematics import LEGS


# 每行：髋、大腿、小腿；每对数值：最小/最大最终舵机角（已包含CAL）。
# 相对P1500分别为(-30,+30)、(-60,+30)、(-60,+60)度。
SERVO_LIMITS_DEG = (
    (60., 120.),
    (30., 120.),
    (30., 150.),
)


def geometric_limits(leg_id):
    """返回指定腿的(3,2)几何角限位，单位弧度，始终按下限/上限排序。"""
    servo = np.asarray(SERVO_LIMITS_DEG, dtype=float)
    if (servo.shape != (3, 2) or not np.isfinite(servo).all()
            or np.any(servo[:, 0] >= servo[:, 1])
            or np.any(servo < 0) or np.any(servo > 180)):
        raise ValueError('舵机限位须为3组0..180度内的有限递增区间')
    leg = LEGS[leg_id]
    result = np.empty((3, 2))
    result[0] = 90. - servo[0, ::-1]  # 髋舵机角方向与几何角相反。
    result[1] = servo[1] - leg.femur_cal - 90.
    result[2] = servo[2] - leg.tibia_cal - 90.
    return np.deg2rad(result)
