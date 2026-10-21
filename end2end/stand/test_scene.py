"""场景检查：图集采样、出生点、高度、合法关节角与随机种子复现。"""
import unittest

import numpy as np

from .env import StandEnv
from .scene import SceneConfig
from .task import StandTask
from hexapod_sim.model import controlled_qpos, controlled_joint_ranges
from hexapod_sim.terrain import ATLAS_TYPES, ATLAS_TOP_Z


class SceneTests(unittest.TestCase):
    def test_pre_settle_starts_formal_episode_from_rest(self):
        env = StandEnv(scene_config=SceneConfig(flat_probability=0.))
        for seed in range(10):
            state = env.reset(seed)
            self.assertTrue(np.all(np.abs(env.errors.servo_deg) <= 3))
            self.assertEqual(env.data.time, 0.)
            np.testing.assert_array_equal(env.data.qvel, 0.)
            self.assertTrue(state['reset_landing_detected'])
            self.assertGreaterEqual(state['reset_settle_duration_s'],
                                    env.scene_config.settle_seconds)
            self.assertTrue(np.isfinite(state['tilt_deg']))

    def test_initial_joint_disturbance_and_handoff(self):
        env = StandEnv(scene_config=SceneConfig())
        env.reset(42)
        bias = env.errors.joint_targets(np.zeros(18))
        np.testing.assert_allclose(env.data.ctrl, env.command + bias, atol=1e-12)
        self.assertTrue(np.any(np.abs(env.initial_joint_offset) > 1e-6))
        self.assertTrue(np.all(np.abs(np.rad2deg(env.initial_joint_offset))
                               <= env.scene_config.initial_joint_deg + 1e-9))
        original = env.command.copy()
        initial_action = env.command_to_action(original)
        env.step(initial_action)
        np.testing.assert_allclose(env.command, original, atol=1e-12)
        env.step(np.zeros(18))  # 模拟策略要求参考角，初始大扰动不再附加。
        self.assertLess(np.linalg.norm(env.command - env.reference), np.linalg.norm(original - env.reference))
        np.testing.assert_allclose(env.data.ctrl - env.command, bias, atol=1e-12)

    def test_reproducible(self):
        env = StandEnv(scene_config=SceneConfig(flat_probability=0.))
        env.reset(42)
        first = env.data.qpos.copy()
        normal = env.ground_normal.copy()
        env.reset(42)
        np.testing.assert_array_equal(first, env.data.qpos)
        np.testing.assert_array_equal(normal, env.ground_normal)
        self.assertIn(env.patch['kind'], ATLAS_TYPES)
        self.assertLessEqual(env.slope_deg, env.scene_config.max_slope_deg)
        self.assertTrue(np.all(np.abs(env.errors.servo_deg) <= 3))
        actual = controlled_qpos(env.model, env.data)
        limits = controlled_joint_ranges(env.model)
        self.assertTrue(np.all(actual >= limits[:, 0]))
        self.assertTrue(np.all(actual <= limits[:, 1]))
        env.reset(43)
        self.assertFalse(np.array_equal(first, env.data.qpos))

    def test_atlas_spawn_and_contacts(self):
        env = StandEnv(scene_config=SceneConfig(flat_probability=0.))
        for seed in range(10):
            state = env.reset(seed)
            n = env.ground_normal
            np.testing.assert_allclose(env.data.geom('floor').xmat.reshape(3, 3)[:, 2], [0, 0, 1], atol=1e-12)
            self.assertAlmostEqual(np.linalg.norm(n), 1.)
            self.assertGreaterEqual(state['ground_height_m'], ATLAS_TOP_Z - .02)
            self.assertTrue(state['on_platform'])
            self.assertEqual(state['terrain_type'], env.patch['kind'])
            self.assertEqual(state['terrain_level'], env.patch['level'])
            self.assertTrue(np.all(np.abs(env.data.qpos[:2] - env.patch['patch_center'])
                                   < env.patch['patch_half_extent']))
            self.assertAlmostEqual(state['clearance_m'], state['height_m'] - state['ground_height_m'])
            for contact in env.data.contact:
                a, b = contact.geom1, contact.geom2
                if env.model.geom_group[a] == 2 or env.model.geom_group[b] == 2:
                    # 预稳定后的柔性接触允许少量稳态压入，但不能出现明显穿模。
                    self.assertGreaterEqual(contact.dist, -10e-3)

    def test_flat_control(self):
        env = StandEnv(scene_config=SceneConfig(1., 0., 0., 0))
        state = env.reset(42)
        self.assertEqual(state['slope_deg'], 0.)
        self.assertAlmostEqual(state['height_m'] - env.support_point[2], state['clearance_m'])
        np.testing.assert_allclose(env.data.ctrl, env.errors.joint_targets(env.command))

    def test_task_uses_local_body_clearance(self):
        task = StandTask(env=StandEnv(scene_config=SceneConfig(1., 0., 0., 0)))
        task.reset(42)
        # 在平坦图集块内移动并补偿地形高度：相对高度保持。
        env = task.env
        clearance = env.state()['clearance_m']
        env.data.qpos[0] += .2
        env.data.qpos[2] = env.ground_height(*env.data.qpos[:2]) + clearance
        self.assertAlmostEqual(env.state()['clearance_m'], clearance)
        _, _, _, _, info = task.step(np.zeros(18))
        self.assertTrue(np.isfinite(info['reward_rates']['body_clearance']))


if __name__ == '__main__':
    unittest.main()
