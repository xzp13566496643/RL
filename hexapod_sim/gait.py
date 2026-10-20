"""三足步态：沿用Arduino相位和足端轨迹逻辑，调用IK输出18个几何关节目标角。"""
import math
import numpy as np

from .kinematics import LEGS, inverse


class Tripod:
    def __init__(self):
        self.tick = 0
        self.cases = [1, 2, 1, 2, 1, 2]

    def step(self, x=40, y=0, r=0, h=0):
        home = np.array([leg.home for leg in LEGS]) * 1000
        if max(abs(x), abs(y), abs(r)) < 10:
            self.tick = 0
            feet = home
        else:
            # Firmware integer division truncates toward zero.
            sx, sy, sr = [math.trunc(a * b / 127) for a, b in ((90, x), (90, y), (35, r))]
            # Match Arduino compute_strides(): saturation after scaling.
            sx, sy, sr = (max(-limit, min(limit, value))
                          for value, limit in zip((sx, sy, sr), (90, 90, 35)))
            sr = math.radians(sr)
            feet = home.copy()
            phase = math.pi * self.tick / 20
            for i, leg in enumerate(LEGS):
                tx, ty = [math.trunc(v * 1000) for v in np.asarray(leg.mount[:2]) + leg.home[:2]]
                rx = ty * math.sin(sr) + tx * math.cos(sr) - tx
                ry = ty * math.cos(sr) - tx * math.sin(sr) - ty
                ax, ay = np.clip([(sx + rx) / 2, (sy + ry) / 2], -50, 50)
                az = np.clip(10 + math.hypot(sx + rx, sy + ry) / 4 + h, 5, 25)
                swing = self.cases[i] == 1
                feet[i, :2] += (-1 if swing else 1) * np.array([ax, ay]) * math.cos(phase)
                if swing:
                    feet[i, 2] += az * math.sin(phase)
                if self.tick >= 19:
                    self.cases[i] = 3 - self.cases[i]
            self.tick = (self.tick + 1) % 20
        return np.concatenate([inverse(i, np.asarray(leg.mount) + feet[i] / 1000)
                               for i, leg in enumerate(LEGS)])
