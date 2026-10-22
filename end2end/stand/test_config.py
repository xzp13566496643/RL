"""训练总配置检查：默认值、奖励配置与可调网络结构。"""
import unittest

import torch

from .actor import StandActor
from .config import DEFAULT_CONFIG, NetworkConfig, PPOConfig, RolloutConfig, TrainingConfig
from .critic import StandCritic
from .task import RewardConfig, TaskConfig, reward_terms


class ConfigTests(unittest.TestCase):
    def test_default_config_matches_current_pipeline(self):
        cfg = DEFAULT_CONFIG
        self.assertEqual(cfg.rollout.num_steps, 512)
        self.assertEqual(cfg.rollout.num_envs, 1)
        self.assertEqual(cfg.network.actor_hidden_dims, (64, 64))
        self.assertEqual(cfg.network.critic_hidden_dims, (256, 128, 64))
        self.assertEqual(cfg.task.reward, RewardConfig())

    def test_network_settings_are_applied(self):
        cfg = NetworkConfig((32,), (48, 24), -1., 'elu')
        actor = StandActor(cfg.initial_log_std, cfg.actor_hidden_dims, cfg.activation)
        critic = StandCritic(cfg.critic_hidden_dims, cfg.activation)
        self.assertEqual(actor(torch.zeros(81)).shape, (18,))
        self.assertEqual(critic(torch.zeros(149)).shape, ())

    def test_reward_settings_are_applied(self):
        reward = RewardConfig(ground_alignment=4.)
        terms = reward_terms(0, .0792, .0792, .02, torch.zeros(6),
                             0, 0, 0, 0, 0, reward)
        self.assertEqual(terms['ground_alignment'], 4.)

    def test_invalid_configs(self):
        with self.assertRaises(ValueError):
            NetworkConfig(actor_hidden_dims=())
        with self.assertRaises(ValueError):
            RolloutConfig(gamma=2.)
        with self.assertRaises(ValueError):
            PPOConfig(epochs=0)
        with self.assertRaises(ValueError):
            TrainingConfig(device='gpu')
        with self.assertRaises(ValueError):
            TaskConfig(reward=RewardConfig(clearance_falloff_m=0.))


if __name__ == '__main__':
    unittest.main()
