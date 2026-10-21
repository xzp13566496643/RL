"""单环境PPO轨迹采集器：连接Gym环境、Actor、Critic与RolloutBuffer。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .rollout import RolloutBuffer


@dataclass(frozen=True)
class CollectionStats:
    steps: int
    completed_episodes: int
    terminated_episodes: int
    truncated_episodes: int
    episode_returns: tuple[float, ...]
    mean_reward_rates: dict[str, float]
    mean_abs_sampled_action: float
    mean_abs_policy_action: float


class SingleEnvCollector:
    """保持单个环境的当前状态，每次collect填满一整个on-policy rollout。"""

    def __init__(self, env, actor, critic, *, device=None, seed=None):
        self.env = env
        self.actor = actor
        self.critic = critic
        actor_device = next(actor.parameters()).device
        critic_device = next(critic.parameters()).device
        self.device = torch.device(device) if device is not None else actor_device
        if self.device.type == 'cuda' and self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        if actor_device != self.device or critic_device != self.device:
            raise ValueError('Actor、Critic与采集器必须位于同一设备')
        self.next_reset_seed = seed
        self.actor_observation = None
        self.critic_observation = None

    def reset(self) -> None:
        observation, info = self.env.reset(seed=self.next_reset_seed)
        # seed只用于初始化Gym随机序列；后续reset()由该序列产生不同而可复现的回合。
        self.next_reset_seed = None
        self.actor_observation = np.asarray(observation, dtype=np.float32)
        self.critic_observation = np.asarray(info['critic_observation'], dtype=np.float32)

    def collect(self, buffer: RolloutBuffer, *, gamma=None, gae_lambda=None,
                normalize_advantages=None) -> CollectionStats:
        if gamma is None or gae_lambda is None or normalize_advantages is None:
            from .config import DEFAULT_CONFIG
            rollout = DEFAULT_CONFIG.rollout
            gamma = rollout.gamma if gamma is None else gamma
            gae_lambda = rollout.gae_lambda if gae_lambda is None else gae_lambda
            normalize_advantages = (rollout.normalize_advantages
                                    if normalize_advantages is None else normalize_advantages)
        if buffer.num_envs != 1:
            raise ValueError('SingleEnvCollector只接受num_envs=1的buffer')
        if buffer.device != self.device:
            raise ValueError('buffer与网络必须位于同一设备')
        if buffer.position != 0:
            raise RuntimeError('开始采集前buffer必须为空')
        if self.actor_observation is None:
            self.reset()

        completed = terminated_count = truncated_count = 0
        episode_returns = []
        reward_rate_sums = {}
        reward_rate_samples = 0
        sampled_action_abs_sum = policy_action_abs_sum = 0.
        while not buffer.full:
            actor_obs = torch.as_tensor(self.actor_observation, device=self.device).unsqueeze(0)
            critic_obs = torch.as_tensor(self.critic_observation, device=self.device).unsqueeze(0)
            with torch.no_grad():
                action, log_probability, _ = self.actor.sample(actor_obs)
                policy_action = self.actor.deterministic(actor_obs)
                value = self.critic(critic_obs)
            sampled_action_abs_sum += float(action.abs().mean())
            policy_action_abs_sum += float(policy_action.abs().mean())

            next_observation, reward, terminated, truncated, info = self.env.step(
                action[0].cpu().numpy())
            rates = info.get('reward_rates')
            if rates:
                reward_rate_samples += 1
                for name, rate_value in rates.items():
                    reward_rate_sums[name] = (reward_rate_sums.get(name, 0.)
                                              + float(rate_value))
            ended = terminated or truncated
            next_critic_array = info.get('critic_observation')
            if next_critic_array is None:
                # 数值异常没有可靠下一状态，且必定是terminated，不允许bootstrap。
                next_critic_array = np.zeros_like(self.critic_observation)
            next_critic_obs = torch.as_tensor(next_critic_array,
                                              dtype=torch.float32,
                                              device=self.device).unsqueeze(0)
            with torch.no_grad():
                next_value = self.critic(next_critic_obs)

            buffer.add(actor_obs, critic_obs, action, log_probability, value, next_value,
                       torch.tensor([reward], device=self.device),
                       torch.tensor([terminated], device=self.device),
                       torch.tensor([truncated], device=self.device))

            if ended:
                completed += 1
                terminated_count += int(terminated)
                truncated_count += int(truncated)
                episode_returns.append(float(info['episode_return']))
                self.reset()
            else:
                self.actor_observation = np.asarray(next_observation, dtype=np.float32)
                self.critic_observation = np.asarray(next_critic_array, dtype=np.float32)

        buffer.compute_returns_and_advantages(gamma, gae_lambda, normalize_advantages)
        mean_rates = ({name: total / reward_rate_samples
                       for name, total in reward_rate_sums.items()}
                      if reward_rate_samples else {})
        return CollectionStats(buffer.num_steps, completed, terminated_count,
                               truncated_count, tuple(episode_returns), mean_rates,
                               sampled_action_abs_sum / buffer.num_steps,
                               policy_action_abs_sum / buffer.num_steps)
