"""自动检查18维动作映射、双向驱动、角度边界、限速和偏置重置可复现性。"""
import unittest

import numpy as np

from .env import StandEnv
from hexapod_sim.joint_limits import SERVO_LIMITS_DEG, geometric_limits
from hexapod_sim.kinematics import LEGS, servo_degrees
from hexapod_sim.model import controlled_qpos, controlled_joint_ranges


class ActionTests(unittest.TestCase):
    def test_passive_foot_pads(self):
        env = StandEnv()
        env.reset(42)
        self.assertEqual(env.model.nu, 18)
        for i in range(6):
            joint = env.model.joint(f'foot{i}_ball_joint')
            self.assertEqual(joint.type[0], 1)  # mjJNT_BALL
            self.assertTrue(joint.limited[0])
            self.assertAlmostEqual(np.rad2deg(joint.range[1]), 25.)
            pad = env.model.geom(f'foot{i}_collision')
            self.assertAlmostEqual(pad.size[0], .030)
            self.assertAlmostEqual(pad.size[1], .002)
        # 被动关节加入qpos，但主动读取接口始终只返回18维。
        self.assertEqual(controlled_qpos(env.model, env.data).shape, (18,))

    def test_shared_bias_initial_and_motion(self):
        env = StandEnv()
        env.reset(42)
        bias = env.errors.joint_targets(np.zeros(18))
        self.assertTrue(np.all(np.abs(env.errors.servo_deg) <= 3))
        np.testing.assert_allclose(controlled_qpos(env.model, env.data) - env.command, bias, atol=1e-12)
        np.testing.assert_allclose(env.data.ctrl - env.command, bias, atol=1e-12)
        for _ in range(5):
            env.step(np.full(18, .2))
            np.testing.assert_allclose(env.data.ctrl - env.command, bias, atol=1e-12)

    def test_servo_limit_conversion(self):
        expected = np.asarray(SERVO_LIMITS_DEG)
        for i in range(len(LEGS)):
            limits = geometric_limits(i)
            # 髋是反向映射，因此转换后逐关节重新排序。
            converted = np.stack([servo_degrees(i, limits[:, k]) for k in (0, 1)], axis=1)
            np.testing.assert_allclose(np.sort(converted, axis=1), expected, atol=1e-12)

    def test_model_and_home_limits(self):
        env = StandEnv()
        env.reset()
        expected = np.concatenate([geometric_limits(i) for i in range(6)])
        np.testing.assert_allclose(controlled_joint_ranges(env.model), expected)
        np.testing.assert_allclose(env.model.actuator_ctrlrange, expected)
        # 不能靠静默裁剪HOME来通过测试。
        self.assertTrue(np.all(env.reference >= expected[:, 0]))
        self.assertTrue(np.all(env.reference <= expected[:, 1]))
        np.testing.assert_allclose(env.command, env.reference)

    def test_zero(self):
        env = StandEnv()
        env.reset()
        np.testing.assert_array_equal(env.observation()[60:78], 0.)
        np.testing.assert_array_equal(env.observation()[78:81], 0.)
        env.step(np.zeros(18))
        np.testing.assert_allclose(env.command, env.reference)

    def test_policy_observation_layout(self):
        command = np.array([.2, -.1, .3], dtype=np.float32)
        env = StandEnv(velocity_command=command)
        env.reset(42)
        action = np.linspace(-1., 1., 18, dtype=np.float32)
        env.step(action)
        obs = env.observation()
        self.assertEqual(obs.shape, (81,))
        self.assertEqual(obs.dtype, np.float32)
        self.assertTrue(np.all(np.abs(obs[:60]) <= env.imu.config.observation_clip))
        np.testing.assert_array_equal(obs[60:78], action)
        np.testing.assert_array_equal(obs[78:81], command * env.velocity_command_scale)

    def test_all_joints(self):
        env = StandEnv()
        for j in range(18):
            for sign in (-1, 1):
                env.reset()
                before = env.command.copy()
                action = np.zeros(18)
                action[j] = sign
                env.step(action)
                delta = env.command - before
                self.assertGreater(sign * delta[j], 0)
                self.assertLessEqual(abs(delta[j]), env.max_delta + 1e-12)
                np.testing.assert_allclose(np.delete(delta, j), 0, atol=1e-12)

    def test_limits_and_bias(self):
        env = StandEnv(joint_bias_deg=2, mount_bias_mm=2, length_bias_mm=2)
        env.reset(42)
        errors = env.errors.servo_deg.copy()
        for sign in (1, -1):
            for _ in range(30):
                before = env.command.copy()
                env.step(np.full(18, sign * 100.))
                self.assertTrue(np.all(np.abs(env.command - before) <= env.max_delta + 1e-12))
                self.assertTrue(np.all(env.command >= env.lower - 1e-12))
                self.assertTrue(np.all(env.command <= env.upper + 1e-12))
                self.assertTrue(np.all(env.data.ctrl >= env.model.actuator_ctrlrange[:, 0] - 1e-12))
                self.assertTrue(np.all(env.data.ctrl <= env.model.actuator_ctrlrange[:, 1] + 1e-12))
                for i in range(6):
                    for angles in (env.command, env.data.ctrl):
                        servo = servo_degrees(i, angles[i*3:i*3+3])
                        self.assertTrue(np.all(servo >= np.asarray(SERVO_LIMITS_DEG)[:, 0] - 1e-10))
                        self.assertTrue(np.all(servo <= np.asarray(SERVO_LIMITS_DEG)[:, 1] + 1e-10))
        env.reset(42)
        np.testing.assert_array_equal(errors, env.errors.servo_deg)
        env.reset(43)
        self.assertFalse(np.array_equal(errors, env.errors.servo_deg))

    def test_invalid(self):
        env = StandEnv()
        env.reset()
        for action in (np.zeros(17), np.full(18, np.nan)):
            with self.assertRaises(ValueError):
                env.step(action)


if __name__ == '__main__':
    unittest.main()
