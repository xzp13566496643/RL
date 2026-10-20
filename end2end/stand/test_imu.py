"""IMU检查：支撑/自由落体比力、轴向、零偏与噪声、100 Hz采样和历史接口。"""
import unittest

import mujoco
import numpy as np

from .env import StandEnv
from .imu import IMU, IMUConfig


IDEAL = IMUConfig(accel_noise_std=0., gyro_noise_std=0., accel_bias_max=0., gyro_bias_max=0.)


def fixture(free=False):
    # 始终使用动态物体。无自由度的静态body不适合验证MuJoCo的
    # 加速度计动力学读数；用外力平衡重力来模拟静止受到支撑。
    xml = ('<mujoco><option gravity="0 0 -9.81"/><worldbody>'
           '<body name="test" axisangle="1 0 0 30">'
           '<freejoint/>'
           '<geom type="sphere" size=".1" mass="1"/>'
           '<site name="imu"/></body></worldbody><sensor>'
           '<accelerometer name="imu_accel" site="imu"/>'
           '<gyro name="imu_gyro" site="imu"/></sensor></mujoco>')
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    if not free:
        # xfrc_applied前三项是世界坐标系力，质量1 kg，向上9.81 N。
        data.xfrc_applied[model.body('test').id, 2] = 9.81
    mujoco.mj_forward(model, data)
    return model, data


class IMUTests(unittest.TestCase):
    def test_supported_and_freefall(self):
        for free in (False, True):
            _, data = fixture(free)
            imu = IMU(IDEAL)
            imu.reset(1)
            imu.sample(data)
            latest = imu.observation().reshape(-1, 6)[-1]
            expected = (np.zeros(3) if free else
                        data.site('imu').xmat.reshape(3, 3).T @ [0, 0, 9.81])
            np.testing.assert_allclose(data.qacc[:3], [0, 0, -9.81] if free else 0, atol=1e-10)
            np.testing.assert_allclose(latest[:3], expected, atol=1e-6)
            np.testing.assert_allclose(latest[3:], 0, atol=1e-6)

    def test_bias_constant_and_reproducible(self):
        _, data = fixture()
        imu = IMU(IMUConfig(accel_noise_std=0., gyro_noise_std=0.))
        imu.reset(42)
        bias = imu.bias.copy()
        imu.sample(data)
        first = imu.observation().copy()
        imu.sample(data)
        np.testing.assert_array_equal(first, imu.observation())
        np.testing.assert_array_equal(bias, imu.bias)
        imu.reset(42)
        np.testing.assert_array_equal(bias, imu.bias)
        imu.reset(43)
        self.assertFalse(np.array_equal(bias, imu.bias))

    def test_noise_and_observation_copy(self):
        _, data = fixture()
        imu = IMU(IMUConfig())
        imu.reset(1)
        imu.sample(data)
        first = imu.observation().copy()
        imu.sample(data)
        self.assertFalse(np.array_equal(first[-6:], imu.observation()[-6:]))
        obs = imu.observation()
        obs[:] = 0
        self.assertFalse(np.array_equal(obs, imu.observation()))

    def test_schedule(self):
        env = StandEnv(imu_config=IDEAL)
        env.reset(42)
        for _ in range(5):
            env.step(np.zeros(18))
        self.assertEqual(env.imu.count, 11)  # 初始1帧+100 ms内10帧。
        np.testing.assert_allclose(np.diff(env.imu.timestamps), .01, atol=1e-10)
        self.assertEqual(env.observation().shape, (81,))
        self.assertTrue(np.isfinite(env.observation()).all())


if __name__ == '__main__':
    unittest.main()
