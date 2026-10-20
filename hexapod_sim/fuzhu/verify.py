"""辅助验证：检查六腿HOME、正逆运动学往返一致性，以及不可达目标的拒绝处理。"""
import numpy as np

from ..kinematics import LEGS, frames, inverse, servo_degrees


def main():
    rng = np.random.default_rng(42)
    worst = 0.0
    for i, leg in enumerate(LEGS):
        target = np.asarray(leg.mount) + leg.home
        q = inverse(i, target)
        actual = frames(i, q)['foot'][:3, 3]
        np.testing.assert_allclose(actual, target, atol=1e-12)
        print(f'leg {i} {leg.name:13s} HOME body mm={np.round(target*1000, 2)} '
              f'servo deg={np.round(servo_degrees(i, q), 3)}')
        # Exercise nonzero yaw, elevation, and knee angles on every installation.
        for _ in range(200):
            original = rng.uniform((-0.5, -0.3, -2.2), (0.5, 0.8, -0.5))
            foot = frames(i, original)['foot'][:3, 3]
            solved = inverse(i, foot)
            error = np.linalg.norm(frames(i, solved)['foot'][:3, 3] - foot)
            worst = max(worst, error)
            assert error < 1e-10
            np.testing.assert_allclose(solved, original, atol=1e-10)
        try:
            inverse(i, np.asarray(leg.mount) + [1, 0, 0])
        except ValueError:
            pass
        else:
            raise AssertionError('Unreachable target was accepted')
    print(f'PASS: 6 HOME targets, 1200 FK/IK round trips, unreachable rejection; max error={worst:.3e} m')


if __name__ == '__main__':
    main()
