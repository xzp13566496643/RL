"""物理演示：默认保持随机初始动作；可选HOME基线或单关节测试，尚无学习策略。"""
import argparse
import json
import time

import mujoco.viewer
import numpy as np

from hexapod_sim.perturbations import add_arguments, from_arguments
from .env import StandEnv
from .imu import IMUConfig
from . import scene


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--headless', action='store_true')
    p.add_argument('--baseline-hold', action='store_true', help='显式运行固定HOME名义角基线；默认保持随机初始动作')
    p.add_argument('--seconds', type=float, default=20., help='物理运行时长；有窗口时到时暂停，窗口保留')
    p.add_argument('--joint', type=int, choices=range(18), help='单关节测试索引0..17；不填则保持站姿')
    p.add_argument('--amplitude', type=float, default=.3, help='正弦动作幅度0..1')
    p.add_argument('--control-ms', type=float, default=20.)
    p.add_argument('--max-speed', type=float, default=60., help='目标角变化速率上限，度/秒')
    p.add_argument('--ideal-imu', action='store_true', help='关闭全部IMU噪声及零偏')
    add_arguments(p)
    p.set_defaults(joint_bias_deg=3.)
    scene.add_arguments(p)
    args = p.parse_args()
    from_arguments(args, p)  # 参数验证；实际采样在reset中进行。
    if not np.isfinite(args.seconds) or args.seconds <= 0 or not 0 <= args.amplitude <= 1:
        p.error('seconds必须有限且>0，amplitude必须在0..1')
    try:
        imu_config = (IMUConfig(accel_noise_std=0., gyro_noise_std=0.,
                                accel_bias_max=0., gyro_bias_max=0.) if args.ideal_imu else IMUConfig())
        env = StandEnv(args.control_ms / 1000, args.max_speed,
                       args.joint_bias_deg, args.mount_bias_mm, args.length_bias_mm,
                       imu_config=imu_config, scene_config=scene.from_arguments(args))
        state = env.reset(args.seed)
    except ValueError as exc:
        p.error(str(exc))
    print(json.dumps(dict(seed=args.seed, **env.errors.report(), slope_deg=state['slope_deg'],
                          initial_tilt_deg=state['tilt_deg'],
                          initial_joint_offset_deg=np.rad2deg(env.initial_joint_offset).tolist()), indent=2))
    viewer = None
    try:
        if not args.headless:
            viewer = mujoco.viewer.launch_passive(env.model, env.data)
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = .85, 135, -30
            viewer.cam.lookat[:] = [env.data.qpos[0], env.data.qpos[1], env.data.qpos[2] - .03]
            if env.platform_enabled:
                viewer.cam.distance = 1.1
        initial_action = env.command_to_action(env.command)
        print('Physics running; no learned policy. Mode: ' +
              ('HOME baseline' if args.baseline_hold else 'joint test' if args.joint is not None
               else 'hold sampled initial joint targets'), flush=True)
        wall_start = time.monotonic()
        reason = 'completed'
        first_failure = None
        for _ in range(int(np.ceil(args.seconds / env.control_dt))):
            if viewer and not viewer.is_running():
                reason = 'window_closed'
                break
            action = np.zeros(18) if args.baseline_hold or args.joint is not None else initial_action.copy()
            # 前1秒只保持参考站姿，随后以4秒周期测试选定关节。
            if args.joint is not None and env.data.time >= 1:
                action[args.joint] = args.amplitude * np.sin(2 * np.pi * (env.data.time - 1) / 4)
            state = env.step(action)
            if int(round(env.data.time / env.control_dt)) % max(1, round(1 / env.control_dt)) == 0:
                latest = env.imu.observation().reshape(-1, 6)[-1]
                print(f't={env.data.time:.2f}s IMU accel(m/s²)={latest[:3]} gyro(rad/s)={latest[3:]}', flush=True)
            if viewer:
                viewer.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                   f't={state["time_s"]:.2f}s | tilt={state["tilt_deg"]:.1f} deg | clearance={state["clearance_m"]:.3f} m',
                                   'Fall detected - continuing for inspection' if first_failure else
                                   'No learned policy | close window to exit')])
                viewer.sync()
                time.sleep(max(0., wall_start + env.data.time - time.monotonic()))
            body_geom = env.model.geom('body_collision').id
            body_touch = any(c.dist <= 0 and body_geom in (c.geom1, c.geom2)
                             and env.model.geom_group[c.geom2 if c.geom1 == body_geom else c.geom1] == 2
                             for c in env.data.contact)
            if first_failure is None and body_touch:
                first_failure = dict(time_s=state['time_s'], clearance_m=state['clearance_m'],
                                     tilt_deg=state['tilt_deg'])
                print('Fall detected; demo continues without resetting: ' + json.dumps(first_failure), flush=True)
        print(json.dumps(dict(status=reason, **state, imu_bias=env.imu.bias.tolist(),
                              first_failure=first_failure, imu_samples=env.imu.count,
                              observation_shape=list(env.observation().shape)), indent=2), flush=True)
        if viewer and viewer.is_running():
            print('Duration reached: physics paused at final state; close window or Ctrl+C to exit.', flush=True)
            viewer.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                               f'PAUSED at t={state["time_s"]:.2f}s | tilt={state["tilt_deg"]:.1f} deg',
                               'Inspect with mouse; close window to exit')])
            while viewer.is_running():
                viewer.sync()
                time.sleep(.02)
    except KeyboardInterrupt:
        pass
    finally:
        if viewer:
            viewer.close()
            time.sleep(1)


if __name__ == '__main__':
    main()
