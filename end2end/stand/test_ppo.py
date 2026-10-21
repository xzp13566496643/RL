"""PPO检查：裁剪目标、真实rollout反向传播、参数更新及非法输入。"""
import unittest

import torch

from .actor import StandActor
from .collector import SingleEnvCollector
from .critic import StandCritic
from .gym_env import StandGymEnv
from .config import PPOConfig
from .ppo import PPOUpdater, clipped_policy_loss
from .rollout import RolloutBuffer
from .scene import SceneConfig
from .task import TaskConfig


class PPOTests(unittest.TestCase):
    def test_policy_clipping(self):
        advantages = torch.tensor([1., -1.])
        log_ratio = torch.log(torch.tensor([2., .5]))
        loss, _, fraction = clipped_policy_loss(log_ratio, advantages, .2)
        # 正优势最多取1.2，负优势时ratio=.5取更保守的-.8。
        self.assertAlmostEqual(float(loss), -.2, places=6)
        self.assertEqual(float(fraction), 1.)

    def test_real_rollout_updates_both_networks(self):
        torch.manual_seed(42)
        env = StandGymEnv(scene_config=SceneConfig(1., 0., 0.),
                          task_config=TaskConfig(episode_seconds=.2), joint_bias_deg=0.)
        actor, critic = StandActor(), StandCritic()
        collector = SingleEnvCollector(env, actor, critic, seed=42)
        buffer = RolloutBuffer(16, 1)
        try:
            collector.collect(buffer)
            actor_before = [parameter.detach().clone() for parameter in actor.parameters()]
            critic_before = [parameter.detach().clone() for parameter in critic.parameters()]
            updater = PPOUpdater(actor, critic, PPOConfig(epochs=2, mini_batch_size=8,
                                                           target_kl=None))
            stats = updater.update(buffer)
            self.assertEqual(stats.updates, 4)
            self.assertEqual(stats.epochs_completed, 2)
            for value in vars(stats).values():
                if not isinstance(value, bool):
                    self.assertTrue(torch.isfinite(torch.tensor(value)))
            self.assertTrue(any(not torch.equal(a, b)
                                for a, b in zip(actor_before, actor.parameters())))
            self.assertTrue(any(not torch.equal(a, b)
                                for a, b in zip(critic_before, critic.parameters())))
        finally:
            env.close()

    def test_rejects_unready_or_too_small_buffer(self):
        actor, critic = StandActor(), StandCritic()
        updater = PPOUpdater(actor, critic, PPOConfig(mini_batch_size=2))
        buffer = RolloutBuffer(1, 1)
        with self.assertRaises(RuntimeError):
            updater.update(buffer)

        # 标记为可训练只用于验证批量边界，正常流程由GAE设置ready。
        buffer.ready = True
        with self.assertRaises(ValueError):
            updater.update(buffer)


if __name__ == '__main__':
    unittest.main()
