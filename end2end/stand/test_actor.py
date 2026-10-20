"""Actor检查：网络维度、参数量、探索采样、确定性动作范围和输入校验。"""
import unittest

import torch

from .actor import ACTION_DIM, OBSERVATION_DIM, StandActor, parameter_count


class ActorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.actor = StandActor()

    def test_structure_and_parameter_count(self):
        self.assertEqual(OBSERVATION_DIM, 81)
        self.assertEqual(ACTION_DIM, 18)
        self.assertEqual(parameter_count(self.actor), 10596)

    def test_single_and_batch_forward(self):
        single = self.actor(torch.zeros(OBSERVATION_DIM))
        batch = self.actor(torch.zeros(4, OBSERVATION_DIM))
        self.assertEqual(single.shape, (ACTION_DIM,))
        self.assertEqual(batch.shape, (4, ACTION_DIM))
        self.assertTrue(torch.isfinite(single).all())
        self.assertTrue(torch.isfinite(batch).all())

    def test_training_distribution(self):
        observation = torch.zeros(4, OBSERVATION_DIM)
        distribution = self.actor.distribution(observation)
        action, log_probability, entropy = self.actor.sample(observation)
        self.assertEqual(distribution.mean.shape, (4, ACTION_DIM))
        self.assertEqual(distribution.stddev.shape, (4, ACTION_DIM))
        self.assertEqual(action.shape, (4, ACTION_DIM))
        self.assertEqual(log_probability.shape, (4,))
        self.assertEqual(entropy.shape, (4,))
        self.assertTrue(torch.isfinite(action).all())
        self.assertTrue(torch.isfinite(log_probability).all())
        self.assertTrue(torch.isfinite(entropy).all())

    def test_deterministic_action_is_bounded(self):
        with torch.no_grad():
            self.actor.mean_network[-1].bias.fill_(10.)
        action = self.actor.deterministic(torch.zeros(OBSERVATION_DIM))
        self.assertTrue(torch.all(action <= 1.))
        self.assertTrue(torch.all(action >= -1.))

    def test_invalid_observation(self):
        with self.assertRaises(ValueError):
            self.actor(torch.zeros(80))
        observation = torch.zeros(OBSERVATION_DIM)
        observation[0] = torch.nan
        with self.assertRaises(ValueError):
            self.actor(observation)


if __name__ == '__main__':
    unittest.main()
