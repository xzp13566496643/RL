"""辅助工具：在重力下保持HOME站姿，检查高度、倾斜、跟踪误差和数值稳定性。"""
import argparse
import json
import time

import mujoco
import mujoco.viewer
import numpy as np

from ..model import make_model, controlled_qpos


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--seconds', type=float, default=10)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error('seconds must be positive')
    model, data = make_model(dynamic=True)
    steps = round(args.seconds / model.opt.timestep)
    viewer = None
    heights = []
    max_tilt = 0.0
    max_error = 0.0
    reference = data.ctrl.copy()
    try:
        if not args.headless:
            viewer = mujoco.viewer.launch_passive(model, data)
            viewer.cam.lookat[:] = [0, 0, 0.07]
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 0.85, 135, -35
        for _ in range(steps):
            if viewer and not viewer.is_running():
                break
            start = time.monotonic()
            mujoco.mj_step(model, data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise RuntimeError('Non-finite physics state')
            heights.append(float(data.qpos[2]))
            rotation = data.xmat[model.body('body').id].reshape(3, 3)
            max_tilt = max(max_tilt, float(np.degrees(np.arccos(np.clip(rotation[2, 2], -1, 1)))))
            max_error = max(max_error, float(np.max(np.abs(controlled_qpos(model, data) - reference))))
            if viewer:
                viewer.sync()
                remaining = model.opt.timestep - (time.monotonic() - start)
                if remaining > 0:
                    time.sleep(remaining)
        result = dict(simulated_seconds=float(data.time), total_mass_kg=float(model.body_mass.sum()),
                      final_height_m=float(data.qpos[2]), min_height_m=min(heights, default=float(data.qpos[2])),
                      max_tilt_deg=max_tilt, max_joint_error_deg=float(np.degrees(max_error)),
                      horizontal_drift_m=float(np.linalg.norm(data.qpos[:2])),
                      contacts=int(data.ncon), warnings=int(sum(w.number for w in data.warning)),
                      finite=True, hardware_parameters_measured=False)
        print(json.dumps(result, indent=2), flush=True)
        if result['warnings'] or result['min_height_m'] < 0.06 or max_tilt > 20:
            raise RuntimeError('Standing check failed: inspect printed metrics')
    finally:
        if viewer:
            viewer.close()
            time.sleep(1)


if __name__ == '__main__':
    main()
