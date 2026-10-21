"""Gym接口检查：官方check_env、种子序列、空间、失败/超时与动作接管。"""
import unittest

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import numpy as np

from .gym_env import StandGymEnv
from .scene import SceneConfig
from .task import TaskConfig


class GymTests(unittest.TestCase):
    def test_official_checker(self):
        env = StandGymEnv()
        try:
            check_env(env, skip_render_check=True)
        finally:
            env.close()

    def test_seed_sequence(self):
        env = StandGymEnv()
        try:
            obs1, info1 = env.reset(seed=42)
            obs2, info2 = env.reset()
            obs3, info3 = env.reset(seed=42)
            obs4, info4 = env.reset()
            np.testing.assert_array_equal(obs1, obs3)
            np.testing.assert_array_equal(obs2, obs4)
            self.assertEqual(info1['episode_seed'], info3['episode_seed'])
            self.assertEqual(info2['episode_seed'], info4['episode_seed'])
            self.assertNotEqual(info1['episode_seed'], info2['episode_seed'])
        finally:
            env.close()

    def test_spaces_timeout_and_action(self):
        env = StandGymEnv(scene_config=SceneConfig(1., 0., 0.),
                          task_config=TaskConfig(episode_seconds=.04))
        try:
            with self.assertRaises(gym.error.ResetNeeded):
                env.step(np.zeros(18))
            obs, _ = env.reset(seed=42)
            self.assertEqual(obs.shape, (81,))
            self.assertEqual(obs.dtype, np.float32)
            self.assertTrue(env.observation_space.contains(obs))
            for action in (np.zeros(17), np.full(18, np.nan), np.full(18, 2.)):
                with self.assertRaises(ValueError):
                    env.step(action)
            before = env.task.env.command.copy()
            obs, reward, terminated, truncated, info = env.step(np.full(18, .2, dtype=np.float32))
            self.assertTrue(np.any(env.task.env.command != before))
            self.assertTrue(env.observation_space.contains(obs))
            self.assertIsInstance(reward, float)
            self.assertFalse(terminated or truncated)
            _, _, terminated, truncated, info = env.step(np.zeros(18, dtype=np.float32))
            self.assertFalse(terminated)
            self.assertTrue(truncated)
            self.assertEqual(info['reason'], 'time_limit')
            with self.assertRaises(gym.error.ResetNeeded):
                env.step(np.zeros(18))
        finally:
            env.close()

    def test_failure(self):
        env = StandGymEnv(scene_config=SceneConfig(1., 0., 0.))
        try:
            env.reset(seed=42)
            # 深度放入地面，确保经过一个控制周期的接触解算后机身仍在接触。
            env.task.env.data.qpos[2] = -.1
            import mujoco
            mujoco.mj_forward(env.task.env.model, env.task.env.data)
            _, reward, terminated, truncated, info = env.step(np.zeros(18, dtype=np.float32))
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertEqual(info['reason'], 'body_contact')
            self.assertLess(reward, 0.)
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
