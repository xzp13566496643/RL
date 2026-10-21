"""端到端站立训练参数总表；训练入口只从这里取得可调超参数。"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .imu import IMUConfig
from .scene import SceneConfig
from .task import RewardConfig, TaskConfig


@dataclass(frozen=True)
class NetworkConfig:
    actor_hidden_dims: tuple[int, ...] = (64, 64)
    critic_hidden_dims: tuple[int, ...] = (256, 128, 64)
    initial_log_std: float = -1.8
    activation: str = 'tanh'

    def __post_init__(self):
        for name, dims in (('Actor', self.actor_hidden_dims), ('Critic', self.critic_hidden_dims)):
            if not dims or any(not isinstance(size, int) or size <= 0 for size in dims):
                raise ValueError(f'{name}隐藏层须为非空正整数序列')
        if not np.isfinite(self.initial_log_std):
            raise ValueError('初始log_std必须是有限数')
        if self.activation not in ('tanh', 'elu', 'relu'):
            raise ValueError('激活函数须为tanh、elu或relu')


@dataclass(frozen=True)
class EnvironmentConfig:
    control_dt: float = .02
    max_speed_deg_s: float = 60.
    joint_bias_deg: float = 3.
    mount_bias_mm: float = 0.
    length_bias_mm: float = 0.
    velocity_command: tuple[float, float, float] = (0., 0., 0.)
    velocity_command_scale: tuple[float, float, float] = (2., 2., .25)
    critic_alignment_scale_deg: float = 30.
    critic_joint_velocity_scale_rad_s: float = 10.
    critic_privileged_clip: float = 10.


@dataclass(frozen=True)
class RolloutConfig:
    num_steps: int = 512
    num_envs: int = 1          # 当前采集器只支持1；并行采集器完成后再调大。
    gamma: float = .995
    gae_lambda: float = .95
    normalize_advantages: bool = True

    def __post_init__(self):
        if (not isinstance(self.num_steps, int) or self.num_steps <= 0
                or not isinstance(self.num_envs, int) or self.num_envs <= 0):
            raise ValueError('rollout步数和环境数必须为正整数')
        if not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError('gamma和gae_lambda必须位于0..1')


@dataclass(frozen=True)
class PPOConfig:
    learning_rate: float = 3e-4
    clip_range: float = .2
    value_coefficient: float = .5
    entropy_coefficient: float = .003
    epochs: int = 3
    mini_batch_size: int = 128
    max_gradient_norm: float = 1.
    target_kl: float | None = .01

    def __post_init__(self):
        positive = (self.learning_rate, self.clip_range, self.value_coefficient,
                    self.max_gradient_norm)
        if any(not np.isfinite(value) or value <= 0 for value in positive):
            raise ValueError('PPO学习率、裁剪值、价值系数和梯度上限必须为有限正数')
        if (not isinstance(self.epochs, int) or isinstance(self.epochs, bool)
                or not isinstance(self.mini_batch_size, int)
                or isinstance(self.mini_batch_size, bool)
                or self.epochs <= 0 or self.mini_batch_size <= 0):
            raise ValueError('PPO训练轮数和mini-batch大小必须为正整数')
        if not np.isfinite(self.entropy_coefficient) or self.entropy_coefficient < 0:
            raise ValueError('熵系数必须为有限非负数')
        if self.target_kl is not None and (not np.isfinite(self.target_kl)
                                           or self.target_kl <= 0):
            raise ValueError('target_kl必须为None或有限正数')


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    device: str = 'auto'       # auto、cpu或cuda
    total_iterations: int = 2000
    log_interval: int = 1
    checkpoint_interval: int = 50
    evaluation_interval: int = 50
    plot_interval: int = 10
    checkpoint_dir: str = 'end2end/stand/checkpoints'
    network: NetworkConfig = field(default_factory=NetworkConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    imu: IMUConfig = field(default_factory=lambda: IMUConfig(
        period_s=.01, history=10,
        accel_noise_std=.05, gyro_noise_std=.005,
        accel_bias_max=.1, gyro_bias_max=.01,
        accel_observation_scale=1. / 9.81, gyro_observation_scale=.25,
        observation_clip=100.))
    scene: SceneConfig = field(default_factory=lambda: SceneConfig(
        flat_probability=.3, max_slope_deg=15., initial_joint_deg=20.,
        max_terrain_level=3))
    task: TaskConfig = field(default_factory=lambda: TaskConfig(
        episode_seconds=10., target_body_clearance_m=.0792,
        body_clearance_tolerance_m=.020, failure_penalty=5.,
        reward=RewardConfig(
            ground_alignment=2., alignment_scale_rad=.2,
            body_clearance=1., clearance_falloff_m=.02,
            foot_contact=.5,
            linear_acceleration_penalty=.02, linear_acceleration_scale=9.81,
            angular_acceleration_penalty=.02, angular_acceleration_scale=20.,
            action_change_penalty=.01, nonfoot_collision_penalty=.25)))
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)

    def __post_init__(self):
        if self.device not in ('auto', 'cpu', 'cuda'):
            raise ValueError('device须为auto、cpu或cuda')
        values = (self.total_iterations, self.log_interval, self.checkpoint_interval,
                  self.evaluation_interval, self.plot_interval)
        if any(not isinstance(value, int) or value <= 0 for value in values):
            raise ValueError('训练迭代数、日志和保存间隔必须为正整数')


DEFAULT_CONFIG = TrainingConfig()


def training_config_from_dict(values) -> TrainingConfig:
    """把checkpoint/config.json中的嵌套字典还原为强类型训练配置。"""
    values = dict(values)
    network = NetworkConfig(**values.pop('network'))
    environment = EnvironmentConfig(**values.pop('environment'))
    imu = IMUConfig(**values.pop('imu'))
    scene = SceneConfig(**values.pop('scene'))
    task_values = dict(values.pop('task'))
    task_values['reward'] = RewardConfig(**task_values['reward'])
    task = TaskConfig(**task_values)
    rollout = RolloutConfig(**values.pop('rollout'))
    ppo = PPOConfig(**values.pop('ppo'))
    return TrainingConfig(
        **values, network=network, environment=environment, imu=imu, scene=scene,
        task=task, rollout=rollout, ppo=ppo)
