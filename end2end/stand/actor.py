"""站立策略Actor：81维实机可用观测映射为18维归一化关节动作。"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal


IMU_HISTORY_DIM = 60
PREVIOUS_ACTION_DIM = 18
VELOCITY_COMMAND_DIM = 3
OBSERVATION_DIM = IMU_HISTORY_DIM + PREVIOUS_ACTION_DIM + VELOCITY_COMMAND_DIM
ACTION_DIM = 18
HIDDEN_DIMS = (64, 64)
ACTION_EPSILON = 1e-6


class StandActor(nn.Module):
    """Tick风格的小型MLP Actor；训练时为高斯策略，部署时输出确定性均值。"""

    def __init__(self, initial_log_std: float = -1.8, hidden_dims=HIDDEN_DIMS,
                 activation='tanh'):
        super().__init__()
        hidden_dims = tuple(hidden_dims)
        if not hidden_dims or any(not isinstance(size, int) or size <= 0 for size in hidden_dims):
            raise ValueError('Actor隐藏层须为非空正整数序列')
        activations = {'tanh': nn.Tanh, 'elu': nn.ELU, 'relu': nn.ReLU}
        if activation not in activations:
            raise ValueError(f'Actor激活函数须为{tuple(activations)}之一')
        layers = []
        previous = OBSERVATION_DIM
        for size in hidden_dims:
            layers.extend((nn.Linear(previous, size), activations[activation]()))
            previous = size
        layers.append(nn.Linear(previous, ACTION_DIM))
        self.mean_network = nn.Sequential(*layers)
        # 每个关节独立的状态无关探索标准差；仅训练时使用。
        self.log_std = nn.Parameter(torch.full((ACTION_DIM,), float(initial_log_std)))
        # 初始均值靠近零动作，即HOME；避免未训练网络一开始打到关节限位。
        nn.init.normal_(self.mean_network[-1].weight, std=0.01)
        nn.init.zeros_(self.mean_network[-1].bias)

    @staticmethod
    def _check_observation(observation: torch.Tensor) -> None:
        if observation.ndim < 1 or observation.shape[-1] != OBSERVATION_DIM:
            raise ValueError(f'Actor输入最后一维必须为{OBSERVATION_DIM}')
        if not torch.isfinite(observation).all():
            raise ValueError('Actor输入包含NaN或Inf')

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """返回未裁剪的高斯均值；接受(81,)或(batch,81)。"""
        self._check_observation(observation)
        return self.mean_network(observation)

    def distribution(self, observation: torch.Tensor) -> Normal:
        """构造PPO训练使用的18维独立高斯分布。"""
        mean = self(observation)
        std = self.log_std.clamp(-3.0, 0.0).exp().expand_as(mean)
        return Normal(mean, std)

    def sample(self, observation: torch.Tensor):
        """训练采样，返回tanh动作、修正后的联合log概率和高斯基分布熵。"""
        distribution = self.distribution(observation)
        raw_action = distribution.sample()
        action = torch.tanh(raw_action)
        log_probability = self._squashed_log_probability(distribution, raw_action)
        # tanh后分布没有简单的解析熵；PPO用基高斯熵作为探索强度代理。
        entropy = distribution.entropy().sum(dim=-1)
        return action, log_probability, entropy

    @staticmethod
    def _squashed_log_probability(distribution: Normal,
                                  raw_action: torch.Tensor) -> torch.Tensor:
        """变量替换公式：log p(tanh(u)) = log p(u) - log|da/du|。"""
        # 等价于log(1-tanh(u)^2)，此形式在|u|较大时数值更稳定。
        log_jacobian = 2. * (torch.log(torch.tensor(2., device=raw_action.device,
                                                    dtype=raw_action.dtype))
                             - raw_action - F.softplus(-2. * raw_action))
        return (distribution.log_prob(raw_action) - log_jacobian).sum(dim=-1)

    def log_probability(self, observation: torch.Tensor,
                        action: torch.Tensor) -> torch.Tensor:
        """PPO更新时，根据已执行的[-1,1]动作重新计算当前策略概率。"""
        if action.shape != observation.shape[:-1] + (ACTION_DIM,):
            raise ValueError('动作批次形状必须与观测批次一致且最后一维为18')
        if not torch.isfinite(action).all():
            raise ValueError('动作包含NaN或Inf')
        bounded = action.clamp(-1. + ACTION_EPSILON, 1. - ACTION_EPSILON)
        raw_action = torch.atanh(bounded)
        return self._squashed_log_probability(self.distribution(observation), raw_action)

    @torch.no_grad()
    def deterministic(self, observation: torch.Tensor) -> torch.Tensor:
        """部署推理：对策略均值作tanh，与训练使用相同的动作变换。"""
        return torch.tanh(self(observation))


def parameter_count(actor: StandActor) -> int:
    """返回Actor全部可训练参数数量（包含18个探索log_std）。"""
    return sum(parameter.numel() for parameter in actor.parameters() if parameter.requires_grad)
