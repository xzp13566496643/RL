"""辅助验证：测试键盘命令、地形高度、障碍碰撞，以及各地图出生点的站立稳定性。"""
import math
import unittest

import mujoco
import numpy as np

from ..keyboard import Keyboard
from ..model import make_model
from ..run import ground_height
from ..gait import Tripod
from ..terrain import TERRAINS, STEEP_SLOPES


class SimulationChecks(unittest.TestCase):
    def test_keyboard(self):
        k = Keyboard()
        self.assertEqual(k.poll()[0], (0, 0, 0))
        for key, expected in [('W', (40, 0, 0)), ('S', (-40, 0, 0)),
                              ('A', (0, 40, 0)), ('D', (0, -40, 0)),
                              ('Q', (0, 0, 40)), ('E', (0, 0, -40))]:
            k.on_key(ord(key))
            self.assertEqual(k.poll()[0], expected)
            self.assertEqual(k.poll()[0], expected)  # Latched after release.
        k.on_key(ord('+'))
        self.assertEqual(k.poll()[0], (0, 0, -45))
        k.on_key(ord(' '))
        self.assertEqual(k.poll()[0], (0, 0, 0))
        k.on_key(ord('R'))
        self.assertTrue(k.poll()[1])
        self.assertFalse(k.poll()[1])
        k.on_key(256)
        self.assertTrue(k.poll()[2])

    def test_strength_limits_and_gait_saturation(self):
        k = Keyboard()
        k.on_key(ord('W'))
        for _ in range(40):
            k.on_key(ord('+'))
        self.assertEqual(k.poll()[0], (150, 0, 0))
        for _ in range(40):
            k.on_key(ord('-'))
        self.assertEqual(k.poll()[0], (10, 0, 0))
        for axis in range(3):
            for sign in (-1, 1):
                a, b = Tripod(), Tripod()
                first, second = [0, 0, 0], [0, 0, 0]
                first[axis], second[axis] = sign * 127, sign * 150
                for _ in range(40):
                    np.testing.assert_allclose(a.step(*first), b.step(*second), atol=1e-12)

    def test_ramp_heights(self):
        m, d = make_model(True, 'ramp', slope_deg=5)
        top = 0.6 * math.tan(math.radians(5))
        for x, z in [(0, 0), (0.8, top/2), (1.2, top), (1.75, top/2), (2.2, 0)]:
            d.qpos[0] = x
            mujoco.mj_forward(m, d)
            self.assertAlmostEqual(ground_height(m, d), z, places=6)

    def test_obstacle_heights_and_contacts(self):
        m, d = make_model(True, 'obstacles', obstacle_mm=8)
        for x, z in [(0, 0), (0.5, 0.008), (0.8, 0.012), (1, 0)]:
            d.qpos[0] = x
            mujoco.mj_forward(m, d)
            self.assertAlmostEqual(ground_height(m, d), z, places=6)
        d.qpos[0], d.qpos[2] = 0.5, 0.02
        mujoco.mj_forward(m, d)
        obstacle_id = m.geom('obstacle_1').id
        self.assertTrue(any(obstacle_id in c.geom for c in d.contact))

    def test_steep_ramp_heights(self):
        m, d = make_model(True, 'steep')
        for i, angle in enumerate(STEEP_SLOPES):
            start = 0.5 + 1.25*i
            top = 0.35*math.tan(math.radians(angle))
            for offset, z in ((0.175, top/2), (0.475, top), (0.775, top/2), (1.1, 0)):
                d.qpos[0] = start + offset
                mujoco.mj_forward(m, d)
                self.assertAlmostEqual(ground_height(m, d), z, places=6)

    def test_spawn_stands_on_all_maps(self):
        for kind in TERRAINS:
            with self.subTest(terrain=kind):
                m, d = make_model(True, kind)
                for _ in range(500):
                    mujoco.mj_step(m, d)
                self.assertTrue(np.isfinite(d.qpos).all())
                self.assertGreater(d.qpos[2], 0.09)
                self.assertFalse(any(w.number for w in d.warning))

    def test_stair_heights(self):
        m, d = make_model(True, 'stairs')
        for x, z in ((0, 0), (0.625, 0.05), (0.875, 0.10),
                     (1.25, 0.15), (1.625, 0.10), (1.875, 0.05), (2.1, 0)):
            d.qpos[0] = x
            mujoco.mj_forward(m, d)
            self.assertAlmostEqual(ground_height(m, d), z, places=6)


if __name__ == '__main__':
    unittest.main()
