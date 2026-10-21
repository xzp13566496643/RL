"""Rollout检查：数据形状、容量、GAE、失败与超时bootstrap语义。"""
import unittest

import torch

from .actor import ACTION_DIM, OBSERVATION_DIM
from .critic import CRITIC_OBSERVATION_DIM
from .rollout import RolloutBuffer


def add_step(buffer, reward, next_value, terminated=False, truncated=False):
    n = buffer.num_envs
    buffer.add(torch.zeros(n, OBSERVATION_DIM),
               torch.zeros(n, CRITIC_OBSERVATION_DIM),
               torch.zeros(n, ACTION_DIM), torch.zeros(n), torch.zeros(n),
               torch.full((n,), float(next_value)), torch.full((n,), float(reward)),
               torch.full((n,), terminated), torch.full((n,), truncated))


class RolloutTests(unittest.TestCase):
    def test_shapes_and_flatten(self):
        buffer = RolloutBuffer(3, 2)
        for _ in range(3):
            add_step(buffer, 1, 0)
        self.assertTrue(buffer.full)
        with self.assertRaises(RuntimeError):
            add_step(buffer, 1, 0)
        buffer.compute_returns_and_advantages()
        batch = buffer.flattened()
        self.assertEqual(batch['actor_observations'].shape, (6, OBSERVATION_DIM))
        self.assertEqual(batch['critic_observations'].shape, (6, CRITIC_OBSERVATION_DIM))
        self.assertEqual(batch['actions'].shape, (6, ACTION_DIM))
        self.assertAlmostEqual(float(batch['advantages'].mean()), 0., places=6)
        buffer.clear()
        self.assertFalse(buffer.full)

    def test_truncation_bootstraps_but_stops_gae_chain(self):
        buffer = RolloutBuffer(2, 1)
        add_step(buffer, reward=1, next_value=0)
        add_step(buffer, reward=1, next_value=10, truncated=True)
        buffer.compute_returns_and_advantages(gamma=1, gae_lambda=1, normalize=False)
        torch.testing.assert_close(buffer.returns[:, 0], torch.tensor([12., 11.]))

    def test_termination_does_not_bootstrap(self):
        buffer = RolloutBuffer(2, 1)
        add_step(buffer, reward=1, next_value=0)
        add_step(buffer, reward=1, next_value=10, terminated=True)
        buffer.compute_returns_and_advantages(gamma=1, gae_lambda=1, normalize=False)
        torch.testing.assert_close(buffer.returns[:, 0], torch.tensor([2., 1.]))

    def test_validation(self):
        buffer = RolloutBuffer(1, 1)
        with self.assertRaises(ValueError):
            buffer.add(torch.zeros(1, OBSERVATION_DIM - 1),
                       torch.zeros(1, CRITIC_OBSERVATION_DIM),
                       torch.zeros(1, ACTION_DIM), torch.zeros(1), torch.zeros(1),
                       torch.zeros(1), torch.zeros(1), torch.zeros(1, dtype=torch.bool),
                       torch.zeros(1, dtype=torch.bool))
        with self.assertRaises(RuntimeError):
            buffer.compute_returns_and_advantages()


if __name__ == '__main__':
    unittest.main()
