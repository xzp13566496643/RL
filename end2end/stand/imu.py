"""100 Hz六轴IMU：原生传感器比力/角速度、每局固定零偏和逐采样白噪声。"""
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IMUConfig:
    period_s: float = .01
    history: int = 10
    # 暂定仿真参数，不是实测值。std是每采样点标准差，不是噪声密度。
    accel_noise_std: float = .05       # m/s²
    gyro_noise_std: float = .005       # rad/s
    accel_bias_max: float = .1         # m/s²，每轴均匀采样±幅度
    gyro_bias_max: float = .01         # rad/s，每轴均匀采样±幅度
    # 参考Eigenbot的固定物理量尺度：先乘scale，再由完整策略观测做异常裁剪。
    accel_observation_scale: float = 1. / 9.81  # 静止重力约映射为1
    gyro_observation_scale: float = .25         # rad/s乘0.25
    observation_clip: float = 100.0             # 仅作数值异常保护

    def __post_init__(self):
        if not np.isfinite(self.period_s) or self.period_s <= 0:
            raise ValueError('IMU采样周期必须为有限正数')
        if not isinstance(self.history, int) or self.history < 1:
            raise ValueError('IMU历史长度必须为正整数')
        params = (self.accel_noise_std, self.gyro_noise_std,
                  self.accel_bias_max, self.gyro_bias_max)
        if any(not np.isfinite(v) or v < 0 for v in params):
            raise ValueError('IMU噪声标准差和零偏幅度必须为有限非负数')
        scales = (self.accel_observation_scale, self.gyro_observation_scale,
                  self.observation_clip)
        if any(not np.isfinite(v) or v <= 0 for v in scales):
            raise ValueError('IMU观测尺度和裁剪范围必须为有限正数')


class IMU:
    def __init__(self, config):
        self.config = config

    def reset(self, seed):
        bias_seed, noise_seed = np.random.SeedSequence(seed).spawn(2)
        bias_rng = np.random.default_rng(bias_seed)
        self.rng = np.random.default_rng(noise_seed)
        limits = np.array([self.config.accel_bias_max] * 3 + [self.config.gyro_bias_max] * 3)
        self.bias = bias_rng.uniform(-limits, limits)
        self.std = np.array([self.config.accel_noise_std] * 3 + [self.config.gyro_noise_std] * 3)
        self.samples = deque(maxlen=self.config.history)
        self.timestamps = deque(maxlen=self.config.history)
        self.count = 0

    def sample(self, data):
        ideal = np.concatenate([data.sensor('imu_accel').data, data.sensor('imu_gyro').data])
        measured = ideal + self.bias + self.rng.normal(size=6) * self.std
        self.samples.append(measured.copy())
        self.timestamps.append(float(data.time))
        self.count += 1

    def observation(self):
        # 启动时复制首个读数补齐历史，不伪造额外采样，不重复采样噪声。
        history = [self.samples[0]] * (self.config.history - len(self.samples)) + list(self.samples)
        return np.asarray(history, dtype=np.float32).reshape(-1).copy()

    def policy_observation(self):
        """返回归一化并裁剪后的IMU历史，供策略使用；原始读数仍由observation提供。"""
        values = self.observation().reshape(-1, 6)
        scales = np.array([self.config.accel_observation_scale] * 3
                          + [self.config.gyro_observation_scale] * 3, dtype=np.float32)
        return np.clip(values * scales, -self.config.observation_clip,
                       self.config.observation_clip).reshape(-1).astype(np.float32)
