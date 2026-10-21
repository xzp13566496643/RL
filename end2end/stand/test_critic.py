"""Critic检查：特权观测组成、网络维度、参数量和输入校验。"""
import unittest

import numpy as np
import torch

from .critic import CRITIC_OBSERVATION_DIM, StandCritic, parameter_count
from .gym_env import StandGymEnv


class CriticTests(unittest.TestCase):
    def test_privileged_observation(self):
        env = StandGymEnv()
        try:
            actor_obs, info = env.reset(seed=42)
            critic_obs = info['critic_observation']
            self.assertEqual(actor_obs.shape, (81,))
            self.assertEqual(critic_obs.shape, (CRITIC_OBSERVATION_DIM,))
            np.testing.assert_array_equal(critic_obs[:81], actor_obs)
            self.assertTrue(np.isfinite(critic_obs).all())
            actor_obs, _, _, _, info = env.step(np.zeros(18, dtype=np.float32))
            np.testing.assert_array_equal(info['critic_observation'][:81], actor_obs)
        finally:
            env.close()

    def test_network(self):
        critic = StandCritic()
        self.assertEqual(parameter_count(critic), 79617)
        self.assertEqual(critic(torch.zeros(CRITIC_OBSERVATION_DIM)).shape, ())
        values = critic(torch.zeros(4, CRITIC_OBSERVATION_DIM))
        self.assertEqual(values.shape, (4,))
        self.assertTrue(torch.isfinite(values).all())

    def test_privileged_values_are_normalized(self):
        env = StandGymEnv()
        try:
            _, info = env.reset(seed=42)
            privileged = info['critic_observation'][81:]
            self.assertTrue(np.all(np.abs(privileged) <= 10.))
            # 初始关节角和PD目标按各自完整安全范围映射，正常均在[-1,1]。
            self.assertTrue(np.all(np.abs(privileged[8:26]) <= 1.))
            self.assertTrue(np.all(np.abs(privileged[44:62]) <= 1.))
        finally:
            env.close()

    def test_invalid_observation(self):
        critic = StandCritic()
        with self.assertRaises(ValueError):
            critic(torch.zeros(CRITIC_OBSERVATION_DIM - 1))
        observation = torch.zeros(CRITIC_OBSERVATION_DIM)
        observation[0] = torch.nan
        with self.assertRaises(ValueError):
            critic(observation)


if __name__ == '__main__':
    unittest.main()
