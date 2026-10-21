"""Gymnasium适配：81维策略观测、18维动作及站立任务五返回值接口。

只适配接口，不引入策略、自动HOME控制、额外观测或奖励修改。
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .env import StandEnv
from .imu import IMUConfig
from .scene import SceneConfig
from .task import StandTask, TaskConfig


class StandGymEnv(gym.Env):
    metadata = {'render_modes': []}  # 此接口只做无窗口训练；现有demo独立负责可视化。

    def __init__(self, *, task_config=None, scene_config=None, imu_config=None,
                 control_dt=.02, max_speed_deg_s=60.,
                 joint_bias_deg=3., mount_bias_mm=0., length_bias_mm=0.,
                 velocity_command=(0., 0., 0.), velocity_command_scale=(2., 2., .25),
                 critic_alignment_scale_deg=30., critic_joint_velocity_scale_rad_s=10.,
                 critic_privileged_clip=10.,
                 render_mode=None):
        super().__init__()
        if render_mode is not None:
            raise ValueError('当前Gym接口仅支持render_mode=None；可视化请使用demo')
        self.render_mode = None
        imu_config = imu_config if imu_config is not None else IMUConfig()
        task_config = task_config if task_config is not None else TaskConfig()
        physics = StandEnv(control_dt=control_dt, max_speed_deg_s=max_speed_deg_s,
                           joint_bias_deg=joint_bias_deg,
                           mount_bias_mm=mount_bias_mm, length_bias_mm=length_bias_mm,
                           imu_config=imu_config,
                           scene_config=scene_config if scene_config is not None else SceneConfig(),
                           velocity_command=velocity_command,
                           velocity_command_scale=velocity_command_scale,
                           critic_target_clearance_m=task_config.target_body_clearance_m,
                           critic_clearance_scale_m=task_config.body_clearance_tolerance_m,
                           critic_alignment_scale_deg=critic_alignment_scale_deg,
                           critic_joint_velocity_scale_rad_s=critic_joint_velocity_scale_rad_s,
                           critic_privileged_clip=critic_privileged_clip)
        self.task = StandTask(physics, task_config)
        # 归一化IMU历史 + 上一动作18维 + 目标速度(vx,vy,wz)3维。
        self.observation_space = spaces.Box(low=-imu_config.observation_clip,
                                            high=imu_config.observation_clip,
                                            shape=(imu_config.history * 6 + 18 + 3,), dtype=np.float32)
        # 18个归一化值由底层映射为绝对目标角，并限幅/限速、施加零位偏置。
        self.action_space = spaces.Box(low=-1., high=1., shape=(18,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options:
            raise ValueError('当前reset不支持非空options；场景参数通过构造函数配置')
        # 使用Gym RNG派生每局种子：首次seed固定后，后续reset()继续抽取新局。
        # 重复reset(seed=42)复现同一局，也可复现整段回合序列。
        episode_seed = int(self.np_random.integers(0, 2**32))
        obs, info = self.task.reset(seed=episode_seed)
        info['episode_seed'] = episode_seed
        return self._observation(obs), info

    def step(self, action):
        if self.task.done:
            raise gym.error.ResetNeeded('请在首次step之前或回合结束后调用reset()')
        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_space.shape or not np.isfinite(action).all():
            raise ValueError('动作必须是18维有限数值数组')
        if not self.action_space.contains(action):
            raise ValueError('Gym动作须在[-1,1]；由策略/训练器在进入环境前限制范围')
        # 唯一动作来源是调用方传入的action，不插入任何固定HOME覆盖命令。
        obs, reward, terminated, truncated, info = self.task.step(action)
        return self._observation(obs), float(reward), bool(terminated), bool(truncated), info

    def _observation(self, obs):
        obs = np.asarray(obs, dtype=np.float32).copy()
        if obs.shape != self.observation_space.shape or not np.isfinite(obs).all():
            raise RuntimeError('策略观测维度不匹配或出现非有限值')
        return obs

    def close(self):
        # 无窗口/后台线程；结束当前回合，可再次reset并复用对象。
        self.task.done = True
