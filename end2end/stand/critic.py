"""PPO训练专用Critic：149维含仿真特权状态的观测映射为一个状态价值。"""
from __future__ import annotations

import torch
from torch import nn


CRITIC_OBSERVATION_DIM = 149
CRITIC_HIDDEN_DIMS = (256, 128, 64)


class StandCritic(nn.Module):
    """非对称Critic；只参与仿真训练，不导出到ESP32。"""

    def __init__(self, hidden_dims=CRITIC_HIDDEN_DIMS, activation='tanh'):
        super().__init__()
        hidden_dims = tuple(hidden_dims)
        if not hidden_dims or any(not isinstance(size, int) or size <= 0 for size in hidden_dims):
            raise ValueError('Critic隐藏层须为非空正整数序列')
        activations = {'tanh': nn.Tanh, 'elu': nn.ELU, 'relu': nn.ReLU}
        if activation not in activations:
            raise ValueError(f'Critic激活函数须为{tuple(activations)}之一')
        layers = []
        previous = CRITIC_OBSERVATION_DIM
        for size in hidden_dims:
            layers.extend((nn.Linear(previous, size), activations[activation]()))
            previous = size
        layers.append(nn.Linear(previous, 1))
        self.value_network = nn.Sequential(*layers)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """接受(149,)或(batch,149)，返回标量或(batch,)价值预测。"""
        if observation.ndim < 1 or observation.shape[-1] != CRITIC_OBSERVATION_DIM:
            raise ValueError(f'Critic输入最后一维必须为{CRITIC_OBSERVATION_DIM}')
        if not torch.isfinite(observation).all():
            raise ValueError('Critic输入包含NaN或Inf')
        return self.value_network(observation).squeeze(-1)


def parameter_count(critic: StandCritic) -> int:
    return sum(parameter.numel() for parameter in critic.parameters() if parameter.requires_grad)
