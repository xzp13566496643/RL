"""单环境采集检查：真实环境数据入buffer、跨回合重置与确定性复现。"""
import unittest

import torch

from .actor import StandActor
from .collector import SingleEnvCollector
from .critic import StandCritic
from .gym_env import StandGymEnv
from .rollout import RolloutBuffer
from .scene import SceneConfig
from .task import TaskConfig


class CollectorTests(unittest.TestCase):
    def make_collection(self, seed):
        torch.manual_seed(seed)
        env = StandGymEnv(scene_config=SceneConfig(1., 0., 0.),
                          task_config=TaskConfig(episode_seconds=.04),
                          joint_bias_deg=0.)
        actor, critic = StandActor(), StandCritic()
        buffer = RolloutBuffer(5, 1)
        collector = SingleEnvCollector(env, actor, critic, seed=seed)
        stats = collector.collect(buffer, normalize_advantages=False)
        return env, buffer, stats

    def test_collects_and_resets_across_time_limits(self):
        env, buffer, stats = self.make_collection(42)
        try:
            self.assertTrue(buffer.full)
            self.assertTrue(buffer.ready)
            self.assertEqual(stats.steps, 5)
            self.assertEqual(stats.completed_episodes, 2)
            self.assertEqual(stats.terminated_episodes, 0)
            self.assertEqual(stats.truncated_episodes, 2)
            self.assertTrue(torch.isfinite(buffer.advantages).all())
            self.assertTrue(torch.isfinite(buffer.returns).all())
            self.assertTrue(torch.all(buffer.actions > -1.))
            self.assertTrue(torch.all(buffer.actions < 1.))
        finally:
            env.close()

    def test_seed_reproduces_rollout(self):
        env1, buffer1, stats1 = self.make_collection(7)
        env2, buffer2, stats2 = self.make_collection(7)
        try:
            torch.testing.assert_close(buffer1.actor_observations,
                                       buffer2.actor_observations)
            torch.testing.assert_close(buffer1.actions, buffer2.actions)
            torch.testing.assert_close(buffer1.rewards, buffer2.rewards)
            self.assertEqual(stats1, stats2)
        finally:
            env1.close()
            env2.close()

    def test_rejects_nonempty_or_multienv_buffer(self):
        env = StandGymEnv()
        actor, critic = StandActor(), StandCritic()
        collector = SingleEnvCollector(env, actor, critic, seed=1)
        try:
            with self.assertRaises(ValueError):
                collector.collect(RolloutBuffer(2, 2))
            buffer = RolloutBuffer(2, 1)
            buffer.position = 1
            with self.assertRaises(RuntimeError):
                collector.collect(buffer)
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
