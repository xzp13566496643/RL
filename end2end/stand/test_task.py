"""任务检查：奖励方向、超时与失败区分、观测接口、回合重置。"""
import unittest

import mujoco
import numpy as np

from .task import StandTask, TaskConfig, reward_terms


class TaskTests(unittest.TestCase):
    def test_reward_direction(self):
        target, tolerance = .0792, .020
        good = sum(reward_terms(0, target, target, tolerance, 6, 0, 0, 0, 0).values())
        bad_cases = (
            reward_terms(.3, target, target, tolerance, 6, 0, 0, 0, 0),
            reward_terms(0, target + tolerance + .02, target, tolerance, 6, 0, 0, 0, 0),
            reward_terms(0, target, target, tolerance, 3, 0, 0, 0, 0),
            reward_terms(0, target, target, tolerance, 6, 9.81, 0, 0, 0),
            reward_terms(0, target, target, tolerance, 6, 0, 20, 0, 0),
            reward_terms(0, target, target, tolerance, 6, 0, 0, 1, 0),
            reward_terms(0, target, target, tolerance, 6, 0, 0, 0, 1),
        )
        for terms in bad_cases:
            self.assertLess(sum(terms.values()), good)

    def test_clearance_reward_is_symmetric_about_home(self):
        target, tolerance, delta = .0792, .020, .012
        low = reward_terms(0, target - tolerance - delta, target, tolerance, 6, 0, 0, 0, 0)
        high = reward_terms(0, target + tolerance + delta, target, tolerance, 6, 0, 0, 0, 0)
        self.assertAlmostEqual(low['body_clearance'], high['body_clearance'])

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

    def test_large_tilt_is_not_terminal(self):
        task = StandTask()
        task.reset(42)
        # 在高处旋转机身，隔离“大倾角”与真实地形接触。
        task.env.data.qpos[2] = .5
        task.env.data.qpos[3:7] = [np.sqrt(.5), np.sqrt(.5), 0., 0.]
        mujoco.mj_forward(task.env.model, task.env.data)
        _, _, terminated, truncated, info = task.step(np.zeros(18))
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertIsNone(info['reason'])

    def test_body_contact(self):
        task = StandTask()
        task.reset(42)
        task.env.data.qpos[2] = .01
        mujoco.mj_forward(task.env.model, task.env.data)
        self.assertTrue(task._contacts()[0])

    def test_leaving_patch_is_not_a_terminal_boundary(self):
        task = StandTask()
        task.reset(42)
        task.env.data.qpos[:2] = task.env.patch['patch_center'] + [1.1, 0.]
        mujoco.mj_forward(task.env.model, task.env.data)
        self.assertFalse(task.env.state()['on_platform'])
        _, _, terminated, _, info = task.step(np.zeros(18))
        self.assertNotEqual(info['reason'], 'off_platform')
        # 可能因人工瞬移后机身接触地形，但不能由地图边界触发。
        if terminated:
            self.assertEqual(info['reason'], 'body_contact')


if __name__ == '__main__':
    unittest.main()
