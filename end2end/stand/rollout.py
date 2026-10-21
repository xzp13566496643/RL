"""PPO轨迹缓存：保存Actor/Critic交互数据，并计算GAE优势和价值回报。"""
from __future__ import annotations

import torch

from .actor import ACTION_DIM, OBSERVATION_DIM
from .critic import CRITIC_OBSERVATION_DIM


class RolloutBuffer:
    """固定长度、多环境的on-policy缓存；每轮PPO更新后必须clear再重新采样。"""

    def __init__(self, num_steps: int, num_envs: int, device: str | torch.device = 'cpu'):
        if num_steps <= 0 or num_envs <= 0:
            raise ValueError('num_steps和num_envs必须为正整数')
        self.num_steps = int(num_steps)
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        shape = (self.num_steps, self.num_envs)
        self.actor_observations = torch.empty((*shape, OBSERVATION_DIM), device=self.device)
        self.critic_observations = torch.empty((*shape, CRITIC_OBSERVATION_DIM), device=self.device)
        self.actions = torch.empty((*shape, ACTION_DIM), device=self.device)
        self.old_log_probabilities = torch.empty(shape, device=self.device)
        self.old_values = torch.empty(shape, device=self.device)
        self.next_values = torch.empty(shape, device=self.device)
        self.rewards = torch.empty(shape, device=self.device)
        self.terminated = torch.empty(shape, dtype=torch.bool, device=self.device)
        self.truncated = torch.empty(shape, dtype=torch.bool, device=self.device)
        self.advantages = torch.empty(shape, device=self.device)
        self.returns = torch.empty(shape, device=self.device)
        self.clear()

    def clear(self) -> None:
        """清空逻辑长度；底层张量复用，避免每轮重复分配内存。"""
        self.position = 0
        self.ready = False

    @property
    def full(self) -> bool:
        return self.position == self.num_steps

    def add(self, actor_observation, critic_observation, action, old_log_probability,
            old_value, next_value, reward, terminated, truncated) -> None:
        """加入所有环境同一时刻的数据；标量环境也必须保留num_envs维。"""
        if self.full:
            raise RuntimeError('rollout已满，须先计算/训练并clear')
        fields = (
            ('actor_observation', actor_observation, (self.num_envs, OBSERVATION_DIM)),
            ('critic_observation', critic_observation, (self.num_envs, CRITIC_OBSERVATION_DIM)),
            ('action', action, (self.num_envs, ACTION_DIM)),
            ('old_log_probability', old_log_probability, (self.num_envs,)),
            ('old_value', old_value, (self.num_envs,)),
            ('next_value', next_value, (self.num_envs,)),
            ('reward', reward, (self.num_envs,)),
        )
        converted = {}
        for name, value, expected_shape in fields:
            tensor = torch.as_tensor(value, dtype=torch.float32, device=self.device)
            if tensor.shape != expected_shape or not torch.isfinite(tensor).all():
                raise ValueError(f'{name}须为{expected_shape}维有限数据')
            converted[name] = tensor
        flags = {}
        for name, value in (('terminated', terminated), ('truncated', truncated)):
            tensor = torch.as_tensor(value, dtype=torch.bool, device=self.device)
            if tensor.shape != (self.num_envs,):
                raise ValueError(f'{name}须为({self.num_envs},)布尔数据')
            flags[name] = tensor

        index = self.position
        self.actor_observations[index].copy_(converted['actor_observation'])
        self.critic_observations[index].copy_(converted['critic_observation'])
        self.actions[index].copy_(converted['action'])
        self.old_log_probabilities[index].copy_(converted['old_log_probability'])
        self.old_values[index].copy_(converted['old_value'])
        self.next_values[index].copy_(converted['next_value'])
        self.rewards[index].copy_(converted['reward'])
        self.terminated[index].copy_(flags['terminated'])
        self.truncated[index].copy_(flags['truncated'])
        self.position += 1
        self.ready = False

    def compute_returns_and_advantages(self, gamma: float = .995, gae_lambda: float = .95,
                                       normalize: bool = True) -> None:
        """计算GAE；超时可bootstrap，但GAE递推不会跨越reset进入下一局。"""
        if not self.full:
            raise RuntimeError('rollout未收满，不能计算GAE')
        if not 0 <= gamma <= 1 or not 0 <= gae_lambda <= 1:
            raise ValueError('gamma和gae_lambda必须位于0..1')
        gae = torch.zeros(self.num_envs, device=self.device)
        for step in reversed(range(self.num_steps)):
            bootstrap = (~self.terminated[step]).float()
            continue_episode = (~(self.terminated[step] | self.truncated[step])).float()
            delta = (self.rewards[step] + gamma * self.next_values[step] * bootstrap
                     - self.old_values[step])
            gae = delta + gamma * gae_lambda * continue_episode * gae
            self.advantages[step] = gae
        self.returns.copy_(self.advantages + self.old_values)
        if normalize:
            mean = self.advantages.mean()
            std = self.advantages.std(unbiased=False)
            self.advantages.sub_(mean).div_(std.clamp_min(1e-8))
        self.ready = True

    def flattened(self) -> dict[str, torch.Tensor]:
        """按时间×环境展平，供后续PPO随机划分mini-batch。"""
        if not self.ready:
            raise RuntimeError('须先计算returns和advantages')
        count = self.num_steps * self.num_envs
        return {
            'actor_observations': self.actor_observations.reshape(count, OBSERVATION_DIM),
            'critic_observations': self.critic_observations.reshape(count, CRITIC_OBSERVATION_DIM),
            'actions': self.actions.reshape(count, ACTION_DIM),
            'old_log_probabilities': self.old_log_probabilities.reshape(count),
            'old_values': self.old_values.reshape(count),
            'advantages': self.advantages.reshape(count),
            'returns': self.returns.reshape(count),
        }
