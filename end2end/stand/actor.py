"""站立策略Actor：81维实机可用观测映射为18维归一化关节动作。"""
from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal


IMU_HISTORY_DIM = 60
PREVIOUS_ACTION_DIM = 18
VELOCITY_COMMAND_DIM = 3
OBSERVATION_DIM = IMU_HISTORY_DIM + PREVIOUS_ACTION_DIM + VELOCITY_COMMAND_DIM
ACTION_DIM = 18
HIDDEN_DIMS = (64, 64)


class StandActor(nn.Module):
    """Tick风格的小型MLP Actor；训练时为高斯策略，部署时输出确定性均值。"""

    def __init__(self, initial_log_std: float = -1.8):
        super().__init__()
        self.mean_network = nn.Sequential(
            nn.Linear(OBSERVATION_DIM, HIDDEN_DIMS[0]),
            nn.Tanh(),
            nn.Linear(HIDDEN_DIMS[0], HIDDEN_DIMS[1]),
            nn.Tanh(),
            nn.Linear(HIDDEN_DIMS[1], ACTION_DIM),
        )
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
        """训练采样，返回原始动作、联合log概率和分布熵。"""
        distribution = self.distribution(observation)
        action = distribution.sample()
        log_probability = distribution.log_prob(action).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        return action, log_probability, entropy

    @torch.no_grad()
    def deterministic(self, observation: torch.Tensor) -> torch.Tensor:
        """部署推理：使用策略均值并限制到环境动作范围[-1,1]。"""
        return self(observation).clamp(-1.0, 1.0)


def parameter_count(actor: StandActor) -> int:
    """返回Actor全部可训练参数数量（包含18个探索log_std）。"""
    return sum(parameter.numel() for parameter in actor.parameters() if parameter.requires_grad)
