"""站立任务：奖励、失败/超时判定及回合管理；真实状态只用于评分和调试。"""
from dataclasses import dataclass, field

import numpy as np

from .env import StandEnv
from .scene import SceneConfig


@dataclass(frozen=True)
class RewardConfig:
    ground_alignment: float = 1.
    alignment_scale_rad: float = .2
    body_clearance: float = 1.
    clearance_falloff_m: float = .02
    # 仅零速度命令时，按六个足垫底面中心到当地地面的距离连续评分。
    foot_contact: float = 1.5
    foot_distance_scale_m: float = .020
    stationary_command_threshold: float = 1e-4
    linear_acceleration_penalty: float = .02
    linear_acceleration_scale: float = 9.81
    angular_acceleration_penalty: float = .02
    angular_acceleration_scale: float = 20.
    action_change_penalty: float = .01
    nonfoot_collision_penalty: float = .25

    def __post_init__(self):
        positive = (self.ground_alignment, self.alignment_scale_rad, self.body_clearance,
                    self.clearance_falloff_m, self.foot_contact, self.foot_distance_scale_m,
                    self.linear_acceleration_scale, self.angular_acceleration_scale)
        penalties = (self.linear_acceleration_penalty, self.angular_acceleration_penalty,
                     self.action_change_penalty, self.nonfoot_collision_penalty)
        if any(not np.isfinite(value) or value <= 0 for value in positive):
            raise ValueError('奖励幅值和尺度必须为有限正数')
        if any(not np.isfinite(value) or value < 0 for value in penalties):
            raise ValueError('惩罚系数必须为有限非负数')
        if (not np.isfinite(self.stationary_command_threshold)
                or self.stationary_command_threshold < 0):
            raise ValueError('零速度命令阈值必须为有限非负数')


@dataclass(frozen=True)
class TaskConfig:
    episode_seconds: float = 10.
    # HOME机身原点高97.2 mm、机身半厚18 mm，因此目标下表面净空为79.2 mm。
    target_body_clearance_m: float = .0792
    body_clearance_tolerance_m: float = .020
    failure_penalty: float = 5.
    reward: RewardConfig = field(default_factory=RewardConfig)

    def __post_init__(self):
        values = (self.episode_seconds, self.target_body_clearance_m,
                  self.body_clearance_tolerance_m, self.failure_penalty)
        if any(not np.isfinite(v) or v <= 0 for v in values):
            raise ValueError('任务参数必须是有限正数')


def reward_terms(alignment_rad, body_clearance, target_clearance, clearance_tolerance,
                 foot_clearances, target_velocity_norm, linear_acceleration,
                 angular_acceleration, command_speed, leg_contacts, config=None):
    """返回每秒奖励分量；step乘控制周期，使改变控制频率时尺度基本一致。"""
    config = config if config is not None else RewardConfig()
    # 容差带严格以HOME净空为中心：target ± tolerance内均为满分。
    clearance_excess = max(abs(body_clearance - target_clearance) - clearance_tolerance, 0.)
    foot_clearances = np.asarray(foot_clearances, dtype=float)
    if foot_clearances.shape != (6,) or not np.isfinite(foot_clearances).all():
        raise ValueError('足端距离必须是6维有限数组')
    stationary = target_velocity_norm <= config.stationary_command_threshold
    foot_proximity = (float(np.mean(np.exp(
        -(np.abs(foot_clearances) / config.foot_distance_scale_m) ** 2)))
                      if stationary else 0.)
    return dict(
        ground_alignment=config.ground_alignment
                         * float(np.exp(-(alignment_rad / config.alignment_scale_rad) ** 2)),
        body_clearance=config.body_clearance
                       * float(np.exp(-(clearance_excess / config.clearance_falloff_m) ** 2)),
        foot_contact=config.foot_contact * foot_proximity,
        high_frequency_motion=(
            -config.linear_acceleration_penalty
            * (linear_acceleration / config.linear_acceleration_scale) ** 2
            -config.angular_acceleration_penalty
            * (angular_acceleration / config.angular_acceleration_scale) ** 2),
        action_change=-config.action_change_penalty * command_speed ** 2,
        nonfoot_collision=-config.nonfoot_collision_penalty * leg_contacts)


class StandTask:
    """类似Gym的五返回值接口，暂不依赖Gymnasium，也不实现自动重置。"""
    def __init__(self, env=None, config=None):
        self.env = env if env is not None else StandEnv(scene_config=SceneConfig())
        self.config = config if config is not None else TaskConfig()
        self.done = True

    def reset(self, seed=None):
        self.env.reset(seed)
        self.steps = 0
        self.total_reward = 0.
        self.done = False
        self.last_obs = self.env.observation()
        return self.last_obs.copy(), dict(state=self.env.state(),
                                          critic_observation=self.env.critic_observation())

    def _feet(self):
        # 圆盘底面中心比球心更适合估计实际支撑面的滑动。
        return np.array([self.env.data.site(f'foot{i}_contact').xpos.copy() for i in range(6)])

    def _foot_clearances(self):
        """六个足垫底面中心到各自正下方地形的竖直距离；侧面碰撞不参与。"""
        positions = self._feet()
        return np.abs(np.array([
            position[2] - self.env.ground_height(*position[:2])
            for position in positions
        ]))

    def _contacts(self):
        """分类机器人与地形的机身、足端和腿杆接触。"""
        body_touch, feet, legs = self.env.terrain_contacts()
        return body_touch, set(np.flatnonzero(feet)), set(np.flatnonzero(legs))

    def step(self, action):
        if self.done:
            raise RuntimeError('回合未开始或已结束，请先reset')
        env, cfg = self.env, self.config
        before_command = env.command.copy()
        before_velocity = env.data.qvel[:6].copy()
        try:
            state = env.step(action)
        except RuntimeError:
            # 环境的非有限状态不能作为下一帧策略观测。
            self.done = True
            reward = -cfg.failure_penalty
            self.total_reward += reward
            return self.last_obs.copy(), reward, True, False, dict(
                reason='nonfinite', episode_return=self.total_reward)
        self.steps += 1
        body_touch, contacts, leg_contacts = self._contacts()
        acceleration = (env.data.qvel[:6] - before_velocity) / env.control_dt
        terms = reward_terms(np.deg2rad(state['ground_alignment_deg']),
                             state['body_clearance_m'], cfg.target_body_clearance_m,
                             cfg.body_clearance_tolerance_m, self._foot_clearances(),
                             float(np.linalg.norm(env.velocity_command)),
                             float(np.linalg.norm(acceleration[:3])),
                             float(np.linalg.norm(acceleration[3:])),
                             float(np.sqrt(np.mean(((env.command - before_command) / env.control_dt) ** 2))),
                             len(leg_contacts), cfg.reward)
        # 离开所选图集块不再强制终止；地形外仍是正常世界平面。
        # 低高度和大倾角只降低奖励，不提前终止，让策略保留恢复机会。
        reason = 'body_contact' if body_touch else None
        terminated = reason is not None
        truncated = not terminated and self.steps * env.control_dt >= cfg.episode_seconds - 1e-12
        if truncated:
            reason = 'time_limit'
        reward = sum(terms.values()) * env.control_dt - (cfg.failure_penalty if terminated else 0.)
        self.total_reward += reward
        self.done = terminated or truncated
        self.last_obs = env.observation()
        return self.last_obs.copy(), float(reward), terminated, truncated, dict(
            reason=reason, state=state, reward_rates=terms, episode_return=self.total_reward,
            foot_contacts=len(contacts), nonfoot_contacts=len(leg_contacts),
            foot_clearances_m=self._foot_clearances().tolist(),
            critic_observation=env.critic_observation())
