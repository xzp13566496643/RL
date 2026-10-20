"""站立任务：奖励、失败/超时判定及回合管理；真实状态只用于评分和调试。"""
from dataclasses import dataclass

import numpy as np

from .env import StandEnv
from .scene import SceneConfig


@dataclass(frozen=True)
class TaskConfig:
    episode_seconds: float = 10.
    target_height_m: float = .0972  # 无偏置固定角站立测得的参考值，可独立调整。
    min_height_m: float = .04
    max_tilt_deg: float = 60.
    failure_penalty: float = 5.

    def __post_init__(self):
        values = (self.episode_seconds, self.target_height_m, self.min_height_m,
                  self.max_tilt_deg, self.failure_penalty)
        if any(not np.isfinite(v) or v <= 0 for v in values):
            raise ValueError('任务参数必须是有限正数')
        if self.min_height_m >= self.target_height_m or self.max_tilt_deg >= 180:
            raise ValueError('失败高度必须低于目标高度，倾角阈值必须小于180度')


def reward_terms(tilt_rad, height_error, angular_speed, linear_speed,
                 command_speed, foot_slip):
    """返回每秒奖励分量；step乘控制周期，使改变控制频率时尺度基本一致。"""
    return dict(upright=2. * float(np.exp(-(tilt_rad / .2) ** 2)),
                height=float(np.exp(-(height_error / .02) ** 2)),
                angular_motion=-.1 * angular_speed ** 2,
                linear_motion=-.5 * linear_speed ** 2,
                command_motion=-.01 * command_speed ** 2,
                foot_slip=-2. * foot_slip ** 2)


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
        return self.last_obs.copy(), dict(state=self.env.state())

    def _feet(self):
        # 圆盘底面中心比球心更适合估计实际支撑面的滑动。
        return np.array([self.env.data.site(f'foot{i}_contact').xpos.copy() for i in range(6)])

    def _contacts(self):
        """只对地形接触计数，忽略腿与机身等自碰撞。"""
        model, data = self.env.model, self.env.data
        body_geom = model.geom('body_collision').id
        feet = {model.geom(f'foot{i}_collision').id: i for i in range(6)}
        body_touch = False
        touching = set()
        for contact in data.contact:
            if contact.dist > 0:
                continue
            for robot, other in ((contact.geom1, contact.geom2), (contact.geom2, contact.geom1)):
                if model.geom_group[other] != 2:
                    continue
                body_touch |= robot == body_geom
                if robot in feet:
                    touching.add(feet[robot])
        return bool(body_touch), touching

    def step(self, action):
        if self.done:
            raise RuntimeError('回合未开始或已结束，请先reset')
        env, cfg = self.env, self.config
        before_command, before_feet = env.command.copy(), self._feet()
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
        body_touch, contacts = self._contacts()
        foot_velocity = (self._feet() - before_feet) / env.control_dt
        # 图集中六只脚可能同时落在不同法向的砖块/坡面上。站立任务先惩罚
        # 接触足在世界XY平面的滑移，不能再共用机身下方的一条地面法向。
        slip = (float(np.sqrt(np.mean(np.sum(foot_velocity[list(contacts), :2] ** 2, axis=1))))
                if contacts else 0.)
        terms = reward_terms(np.deg2rad(state['tilt_deg']), state['clearance_m'] - cfg.target_height_m,
                             float(np.linalg.norm(env.data.sensor('imu_gyro').data)),
                             float(np.linalg.norm(env.data.qvel[:3])),
                             float(np.sqrt(np.mean(((env.command - before_command) / env.control_dt) ** 2))),
                             slip)
        reason = ('off_platform' if not state['on_platform'] else
                  'body_contact' if body_touch else 'low_height' if state['clearance_m'] < cfg.min_height_m
                  else 'excessive_tilt' if state['tilt_deg'] > cfg.max_tilt_deg else None)
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
            foot_contacts=len(contacts), foot_slip_m_s=slip)
