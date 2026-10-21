"""PPO参数更新器：使用rollout中的优势与回报更新Actor和Critic。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .config import PPOConfig
from .rollout import RolloutBuffer


@dataclass(frozen=True)
class PPOStats:
    updates: int
    epochs_completed: int
    actor_loss: float
    critic_loss: float
    entropy: float
    approximate_kl: float
    clip_fraction: float
    gradient_norm: float
    early_stopped: bool


def clipped_policy_loss(log_ratio: torch.Tensor, advantages: torch.Tensor,
                        clip_range: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回Actor损失、近似KL与触发裁剪的样本比例。"""
    ratio = torch.exp(log_ratio.clamp(-20., 20.))
    unclipped = ratio * advantages
    clipped = ratio.clamp(1. - clip_range, 1. + clip_range) * advantages
    loss = -torch.minimum(unclipped, clipped).mean()
    approximate_kl = ((ratio - 1.) - log_ratio).mean()
    clip_fraction = ((ratio - 1.).abs() > clip_range).float().mean()
    return loss, approximate_kl, clip_fraction


class PPOUpdater:
    """一个优化器共同管理两套独立网络；每次update只使用当前一批on-policy数据。"""

    def __init__(self, actor, critic, config=None):
        self.actor = actor
        self.critic = critic
        self.config = config if config is not None else PPOConfig()
        actor_device = next(actor.parameters()).device
        critic_device = next(critic.parameters()).device
        if actor_device != critic_device:
            raise ValueError('Actor和Critic必须位于同一设备')
        self.device = actor_device
        self.parameters = list(actor.parameters()) + list(critic.parameters())
        self.optimizer = torch.optim.Adam(self.parameters, lr=self.config.learning_rate)

    def update(self, buffer: RolloutBuffer) -> PPOStats:
        if not buffer.ready:
            raise RuntimeError('buffer须先收满并计算GAE')
        if buffer.device != self.device:
            raise ValueError('buffer与网络必须位于同一设备')
        batch = buffer.flattened()
        count = batch['advantages'].shape[0]
        if self.config.mini_batch_size > count:
            raise ValueError('mini_batch_size不能大于rollout样本数')

        totals = dict(actor_loss=0., critic_loss=0., entropy=0., approximate_kl=0.,
                      clip_fraction=0., gradient_norm=0.)
        updates = 0
        epochs_completed = 0
        early_stopped = False
        for _ in range(self.config.epochs):
            permutation = torch.randperm(count, device=self.device)
            epoch_kls = []
            for start in range(0, count, self.config.mini_batch_size):
                indices = permutation[start:start + self.config.mini_batch_size]
                actor_obs = batch['actor_observations'][indices]
                critic_obs = batch['critic_observations'][indices]
                actions = batch['actions'][indices]
                old_log_probability = batch['old_log_probabilities'][indices]
                advantages = batch['advantages'][indices]
                returns = batch['returns'][indices]
                old_values = batch['old_values'][indices]

                new_log_probability = self.actor.log_probability(actor_obs, actions)
                log_ratio = new_log_probability - old_log_probability
                actor_loss, approximate_kl, clip_fraction = clipped_policy_loss(
                    log_ratio, advantages, self.config.clip_range)

                new_values = self.critic(critic_obs)
                clipped_values = old_values + (new_values - old_values).clamp(
                    -self.config.clip_range, self.config.clip_range)
                value_loss = (new_values - returns).square()
                clipped_value_loss = (clipped_values - returns).square()
                critic_loss = .5 * torch.maximum(value_loss, clipped_value_loss).mean()

                entropy = self.actor.distribution(actor_obs).entropy().sum(dim=-1).mean()
                total_loss = (actor_loss + self.config.value_coefficient * critic_loss
                              - self.config.entropy_coefficient * entropy)
                if not torch.isfinite(total_loss):
                    raise RuntimeError('PPO损失出现NaN或Inf')
                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                gradient_norm = nn.utils.clip_grad_norm_(self.parameters,
                                                          self.config.max_gradient_norm)
                if not torch.isfinite(gradient_norm):
                    raise RuntimeError('PPO梯度出现NaN或Inf')
                self.optimizer.step()
                # 与distribution中的有效范围一致，避免参数越界后因clamp而失去梯度。
                with torch.no_grad():
                    self.actor.log_std.clamp_(-3., 0.)

                values = (actor_loss, critic_loss, entropy, approximate_kl,
                          clip_fraction, gradient_norm)
                for name, value in zip(totals, values):
                    totals[name] += float(value.detach())
                epoch_kls.append(float(approximate_kl.detach()))
                updates += 1
            epochs_completed += 1
            if (self.config.target_kl is not None and epoch_kls
                    and np.mean(epoch_kls) > 1.5 * self.config.target_kl):
                early_stopped = True
                break

        means = {name: value / updates for name, value in totals.items()}
        return PPOStats(updates=updates, epochs_completed=epochs_completed,
                        early_stopped=early_stopped, **means)
