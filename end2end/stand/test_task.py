"""任务检查：奖励方向、超时与失败区分、观测接口、回合重置。"""
import unittest

import mujoco
import numpy as np

from .task import StandTask, TaskConfig, reward_terms


class TaskTests(unittest.TestCase):
    def test_reward_direction(self):
        good = sum(reward_terms(0, 0, 0, 0, 0, 0).values())
        for index in range(6):
            values = [0.] * 6
            values[index] = .3
            self.assertLess(sum(reward_terms(*values).values()), good)

    def test_timeout(self):
        task = StandTask(config=TaskConfig(episode_seconds=.04))
        obs, _ = task.reset(42)
        self.assertEqual(obs.shape, (81,))
        _, _, terminated, truncated, _ = task.step(np.zeros(18))
        self.assertFalse(terminated or truncated)
        obs, reward, terminated, truncated, info = task.step(np.zeros(18))
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info['reason'], 'time_limit')
        self.assertTrue(np.isfinite(reward))
        self.assertTrue(np.isfinite(obs).all())
        with self.assertRaises(RuntimeError):
            task.step(np.zeros(18))
        task.reset(42)
        self.assertFalse(task.done)
        self.assertEqual(task.total_reward, 0)

    def test_fallen(self):
        # 人为提高失败高度阈值，稳定验证失败路径，避免穿地引发接触反弹。
        task = StandTask(config=TaskConfig(target_height_m=.2, min_height_m=.15))
        task.reset(42)
        _, reward, terminated, truncated, info = task.step(np.zeros(18))
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertIn(info['reason'], ('body_contact', 'low_height', 'excessive_tilt'))
        self.assertLess(reward, 0)

    def test_body_contact(self):
        task = StandTask()
        task.reset(42)
        task.env.data.qpos[2] = .01
        mujoco.mj_forward(task.env.model, task.env.data)
        self.assertTrue(task._contacts()[0])


if __name__ == '__main__':
    unittest.main()
