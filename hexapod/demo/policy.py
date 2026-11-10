"""仅依赖NumPy的确定性Actor推理。"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class ActorPolicy:
    """加载从MJX checkpoint导出的三层tanh Actor。"""

    def __init__(self, path):
        path = Path(path)
        with np.load(path) as values:
            self.iteration = int(values['iteration'])
            self.total_steps = int(values['total_steps'])
            self.weights = tuple(np.asarray(values[f'actor_w{i}'], dtype=np.float32)
                                 for i in range(3))
            self.biases = tuple(np.asarray(values[f'actor_b{i}'], dtype=np.float32)
                                for i in range(3))
        expected = ((65, 64), (64, 64), (64, 18))
        if tuple(value.shape for value in self.weights) != expected:
            raise ValueError(f'Actor权重维度不匹配：{tuple(v.shape for v in self.weights)}')
        if tuple(value.shape for value in self.biases) != ((64,), (64,), (18,)):
            raise ValueError('Actor偏置维度不匹配')

    def __call__(self, observation):
        value = np.asarray(observation, dtype=np.float32)
        if value.shape != (65,) or not np.isfinite(value).all():
            raise ValueError('Actor观测必须是65维有限数组')
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            value = np.tanh(value @ weight + bias)
        return np.tanh(value @ self.weights[-1] + self.biases[-1]).astype(np.float32)
