"""实机误差：每次启动固定采样零位、髋安装位置和连杆长度偏置，不修改控制侧IK。"""
from dataclasses import dataclass

import numpy as np

from .kinematics import COXA, FEMUR, TIBIA


@dataclass(frozen=True)
class Perturbations:
    # 每行对应一条腿；三列分别为髋/大腿/小腿、机身XYZ、三段连杆。
    servo_deg: np.ndarray
    mount_mm: np.ndarray
    length_mm: np.ndarray

    @classmethod
    def sample(cls, seed=0, joint_deg=0., mount_mm=0., length_mm=0.):
        bounds = (joint_deg, mount_mm, length_mm)
        if any(not np.isfinite(v) or v < 0 for v in bounds):
            raise ValueError('偏置幅度必须为有限非负数')
        if seed < 0:
            raise ValueError('seed必须非负')
        if length_mm >= min(COXA, FEMUR, TIBIA) * 1000:
            raise ValueError('连杆偏置幅度必须小于最短连杆长度42 mm')
        # 三类随机数独立，调整某类幅度不会改变另外两类采样。
        streams = np.random.SeedSequence(seed).spawn(3)
        arrays = [np.random.default_rng(s).uniform(-b, b, (6, 3))
                  for s, b in zip(streams, bounds)]
        for a in arrays:
            a.setflags(write=False)
        return cls(*arrays)

    def joint_targets(self, nominal):
        """在IK之后施加舵机角偏置；Arduino髋角方向与几何角相反。"""
        offset = np.deg2rad(self.servo_deg * [-1, 1, 1]).reshape(18)
        return np.asarray(nominal) + offset

    def report(self):
        return dict(servo_bias_deg=self.servo_deg.tolist(),
                    mount_bias_mm=self.mount_mm.tolist(),
                    length_bias_mm=self.length_mm.tolist())


def add_arguments(parser):
    parser.add_argument('--seed', type=int, default=0, help='固定误差随机种子；R重置不重新采样')
    parser.add_argument('--joint-bias-deg', type=float, default=0., help='18个舵机角偏置均匀采样范围±度')
    parser.add_argument('--mount-bias-mm', type=float, default=0., help='六个髋安装位置XYZ偏置范围±mm')
    parser.add_argument('--length-bias-mm', type=float, default=0., help='每条腿三段长度偏置范围±mm')


def from_arguments(args, parser):
    try:
        return Perturbations.sample(args.seed, args.joint_bias_deg,
                                    args.mount_bias_mm, args.length_bias_mm)
    except ValueError as exc:
        parser.error(str(exc))
