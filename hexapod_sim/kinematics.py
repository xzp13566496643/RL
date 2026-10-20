"""运动学：保存六腿安装、连杆尺寸、HOME与CAL参数，提供共用单腿FK/IK及舵机角映射。"""
from dataclasses import dataclass
import math

import numpy as np

COXA, FEMUR, TIBIA = 0.042, 0.062, 0.120


@dataclass(frozen=True)
class Leg:
    name: str
    mount: tuple
    yaw_deg: float
    home: tuple  # Hip-relative, with axes parallel to body axes (firmware convention).
    femur_cal: float
    tibia_cal: float


LEGS = (
    Leg('left_front', (0.0684, 0.0395, 0), 30, (0.1225, 0.0707, -0.0916), -5, 75),
    Leg('left_middle', (0, 0.079, 0), 90, (0, 0.1413, -0.0916), -2, 75),
    Leg('left_rear', (-0.0684, 0.0395, 0), 150, (-0.1225, 0.0707, -0.0916), -5, 75),
    Leg('right_rear', (-0.0684, -0.0395, 0), 210, (-0.1225, -0.0707, -0.0916), -12, 80),
    Leg('right_middle', (0, -0.079, 0), 270, (0, -0.1413, -0.0916), -15, 83),
    Leg('right_front', (0.0684, -0.0395, 0), 330, (0.1225, -0.0707, -0.0916), -13, 74),
)


def transform(position=(0, 0, 0), yaw=0.0, pitch=0.0):
    """Homogeneous transform Rz(yaw) Ry(-pitch); positive pitch lifts +X."""
    c, s = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    result = np.eye(4)
    result[:3, :3] = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1))) @ np.array(
        ((cp, 0, -sp), (0, 1, 0), (sp, 0, cp)))
    result[:3, 3] = position
    return result


def frames(leg_id, q):
    """Body-to-joint/foot transforms for q=(hip yaw, thigh elevation, knee bend).

    All-zero q is a straight horizontal outward leg. Hip axis is +Z;
    thigh and knee axes are local -Y. The selected IK branch has knee <= 0.
    """
    leg = LEGS[leg_id]
    mount = transform(leg.mount, yaw=math.radians(leg.yaw_deg))
    hip = mount @ transform(yaw=q[0])
    thigh = hip @ transform((COXA, 0, 0), pitch=q[1])
    shin = thigh @ transform((FEMUR, 0, 0), pitch=q[2])
    foot = shin @ transform((TIBIA, 0, 0))
    return dict(mount=mount, hip=hip, thigh=thigh, shin=shin, foot=foot)


def inverse(leg_id, foot_body):
    """Body-frame foot target -> geometric joint radians; reject unreachable targets."""
    leg = LEGS[leg_id]
    mount = transform(leg.mount, yaw=math.radians(leg.yaw_deg))
    x, y, z = mount[:3, :3].T @ (np.asarray(foot_body) - np.asarray(leg.mount))
    radial = math.hypot(x, y) - COXA
    distance = math.hypot(radial, z)
    if not abs(TIBIA - FEMUR) < distance < TIBIA + FEMUR:
        raise ValueError(f'Leg {leg_id}: unreachable target, L3={distance:.6f} m')
    cosine = (distance**2 - FEMUR**2 - TIBIA**2) / (2 * FEMUR * TIBIA)
    knee = -math.acos(max(-1, min(1, cosine)))
    thigh = math.atan2(z, radial) - math.atan2(TIBIA * math.sin(knee),
                                                             FEMUR + TIBIA * math.cos(knee))
    return np.array((math.atan2(y, x), thigh, knee))


def servo_degrees(leg_id, q, cog_femur=0, cog_tibia=0):
    """Firmware hexapod_type=1 mapping, before its 0..180 degree clipping.

    These are command angles, not measured encoder positions or verified limits.
    """
    leg = LEGS[leg_id]
    hip, thigh, knee = np.degrees(q)
    return np.array((90 - hip, thigh + leg.femur_cal + 90 + cog_femur,
                     knee + leg.tibia_cal + 90 + cog_tibia))
