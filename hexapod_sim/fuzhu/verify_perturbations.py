"""辅助验证：检查误差可复现、舵机角方向、实际几何变化及默认模型兼容性。"""
import unittest

import numpy as np

from ..kinematics import LEGS, inverse, COXA
from ..model import make_model, controlled_qpos
from ..perturbations import Perturbations


class PerturbationTests(unittest.TestCase):
    def test_sampling(self):
        a = Perturbations.sample(42, 2, 2, 2)
        b = Perturbations.sample(42, 2, 0, 2)
        np.testing.assert_array_equal(a.servo_deg, b.servo_deg)
        np.testing.assert_array_equal(a.length_mm, b.length_mm)
        np.testing.assert_array_equal(a.mount_mm, Perturbations.sample(42, 2, 2, 2).mount_mm)
        self.assertFalse(a.servo_deg.flags.writeable)
        np.testing.assert_allclose(a.joint_targets(np.zeros(18)).reshape(6, 3),
                                   np.deg2rad(a.servo_deg * [-1, 1, 1]))

    def test_geometry_and_home(self):
        make_model()  # 默认无误差时，模型内部仍验证六足HOME。
        errors = Perturbations.sample(42, 2, 2, 2)
        model, data = make_model(perturbations=errors)
        nominal = np.concatenate([inverse(i, np.asarray(leg.mount) + leg.home)
                                  for i, leg in enumerate(LEGS)])
        expected = np.clip(errors.joint_targets(nominal), model.jnt_range[:, 0], model.jnt_range[:, 1])
        np.testing.assert_allclose(data.qpos, expected)
        for i, leg in enumerate(LEGS):
            np.testing.assert_allclose(model.body(f'leg{i}_mount').pos,
                                       np.asarray(leg.mount) + errors.mount_mm[i] / 1000)
            self.assertAlmostEqual(model.body(f'leg{i}_femur').pos[0],
                                   COXA + errors.length_mm[i, 0] / 1000)
        nominal_foot = np.asarray(LEGS[0].mount) + LEGS[0].home + [0, 0, .0976]
        self.assertGreater(np.linalg.norm(data.site('foot0').xpos - nominal_foot), 1e-6)
        dynamic_model, dynamic = make_model(True, perturbations=errors)
        np.testing.assert_allclose(dynamic.ctrl, controlled_qpos(dynamic_model, dynamic))
        self.assertGreaterEqual(min(dynamic.site(f'foot{i}').xpos[2] for i in range(6)), .009999)

    def test_invalid(self):
        for values in ((0, -1, 0, 0), (0, 0, 0, 42), (-1, 0, 0, 0)):
            with self.assertRaises(ValueError):
                Perturbations.sample(*values)


if __name__ == '__main__':
    unittest.main()
