"""动态仿真主入口：接收固定或键盘命令，调度步态、执行器、物理步进、显示及状态检查。"""
import argparse
import json
import time

import mujoco
import mujoco.viewer
import numpy as np

from .gait import Tripod
from .model import make_model
from .terrain import TERRAINS
from .keyboard import Keyboard, ControlWindow
from .perturbations import add_arguments, from_arguments


def ground_height(model, data):
    geomid = np.zeros(1, dtype=np.int32)
    start = np.array([data.qpos[0], data.qpos[1], 2.0])
    # Terrain is group 2; robot and visual markers are excluded from this ray.
    groups = np.array([0, 0, 1, 0, 0, 0], dtype=np.uint8)
    distance = mujoco.mj_ray(model, data, start, np.array([0., 0., -1.]),
                             groups, True, -1, geomid)
    return 0.0 if distance < 0 else float(start[2] - distance)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--headless', action='store_true')
    p.add_argument('--keyboard', action='store_true', help='Open a separate robot-control window; starts stopped')
    p.add_argument('--seconds', type=float, default=30, help='Duration after 1s settling')
    p.add_argument('--x', type=int, default=40, help='Internal command AFTER BLE x0.5; ignored in keyboard mode')
    p.add_argument('--y', type=int, default=0)
    p.add_argument('--r', type=int, default=0)
    p.add_argument('--send-ms', type=int, default=200)
    p.add_argument('--terrain', choices=TERRAINS, default='flat')
    p.add_argument('--slope-deg', type=float, default=5)
    p.add_argument('--obstacle-mm', type=float, default=8, help='Second bar height is 1.5x this')
    p.add_argument('--overview', action='store_true')
    add_arguments(p)
    args = p.parse_args()
    errors = from_arguments(args, p)
    if not np.isfinite(args.seconds) or args.seconds <= 0 or args.send_ms < 20 or args.send_ms % 20:
        p.error('seconds must be finite and positive; send-ms must be a multiple of 20, >=20')
    if args.keyboard and args.headless:
        p.error('--keyboard requires a viewer')
    if any(abs(v) > 150 for v in (args.x, args.y, args.r)):
        p.error('x/y/r must be in -150..150; gait saturates at magnitude 127')
    if not 0 < args.slope_deg <= 20 or not 0 < args.obstacle_mm <= 100:
        p.error('slope-deg in (0,20], obstacle-mm in (0,100]')
    model, data = make_model(True, args.terrain, args.slope_deg, args.obstacle_mm, errors)
    print(json.dumps(dict(seed=args.seed, **errors.report()), indent=2), flush=True)
    initial = data.qpos.copy()
    home = data.ctrl.copy()
    keyboard = Keyboard()
    command = (0, 0, 0) if args.keyboard else (args.x, args.y, args.r)
    gait = Tripod()
    viewer = None
    control_window = None
    dt = model.opt.timestep
    frame_steps = round(0.02 / dt)
    settle_steps = round(1 / dt)
    settle_until = settle_steps
    duration = settle_steps + round(args.seconds / dt)
    status = 'completed'
    min_clearance, max_tilt = float('inf'), 0.
    start_position = None
    resets = 0
    fallen = False
    wall_start = time.monotonic()
    try:
        if not args.headless:
            viewer = mujoco.viewer.launch_passive(model, data)
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.3, 135, -35
            if args.overview:
                viewer.cam.distance, viewer.cam.elevation = 3.8, -50
                viewer.cam.lookat[:] = [1.1, 0, 0.05]
                if args.terrain == 'steep':
                    viewer.cam.distance = 5.7
                    viewer.cam.lookat[:] = [1.95, 0, 0.08]
                elif args.terrain == 'stairs':
                    viewer.cam.distance = 3.1
                    viewer.cam.lookat[:] = [0.95, 0, 0.08]
        if args.keyboard:
            control_window = ControlWindow(keyboard)
            print('Use the separate robot-control window: W/S forward/back, A/D strafe, Q/E rotate; '
                  'SPACE stop, R reset, +/- strength, ESC exit. Focus loss stops motion.', flush=True)
        for step in range(duration):
            if viewer and not viewer.is_running():
                status = 'window_closed'
                break
            if step % frame_steps == 0 and args.keyboard:
                control_window.update(command, fallen)
                new_command, reset, quit_requested, strength = keyboard.poll()
                if quit_requested:
                    status = 'user_quit'
                    break
                if reset:
                    mujoco.mj_resetData(model, data)
                    data.qpos[:] = initial
                    data.ctrl[:] = home
                    mujoco.mj_forward(model, data)
                    gait = Tripod()
                    settle_until = step + settle_steps
                    resets += 1
                    fallen = False
                    status = 'completed'
                if new_command != command:
                    command = new_command
                    print(f'X/Y/R={command}', flush=True)
            if step == settle_steps:
                start_position = data.qpos[:3].copy()
            walk_step = step - settle_until
            if walk_step >= 0 and walk_step % frame_steps == 0 and not fallen:
                target = gait.step(*command)
                if (walk_step // frame_steps) % (args.send_ms // 20) == 0:
                    data.ctrl[:] = np.clip(errors.joint_targets(target), model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                status = 'nonfinite'
                break
            clearance = data.qpos[2] - ground_height(model, data)
            tilt = float(np.degrees(np.arccos(np.clip(data.xmat[model.body('body').id, 8], -1, 1))))
            if walk_step >= 0:
                min_clearance, max_tilt = min(min_clearance, float(clearance)), max(max_tilt, tilt)
                if clearance < 0.04 or tilt > 60:
                    status = 'fallen'
                    fallen = True
                    if not args.keyboard:
                        break
                    # Keep the viewer open for R reset; do not continue driving a fallen robot.
                    command = (0, 0, 0)
            if viewer and step % frame_steps == 0:
                if not args.overview:
                    viewer.cam.lookat[:] = [*data.qpos[:2], data.qpos[2]]
                if args.keyboard:
                    viewer.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                       f'{args.terrain} | X/Y/R {command} | clearance {clearance:.3f}m | {"FALLEN - R reset" if fallen else "running"}',
                                       'Robot keys: use the separate control window\nMuJoCo viewer keys keep their built-in functions')])
                viewer.sync()
                remaining = wall_start + (step + 1) * dt - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
        warnings = int(sum(w.number for w in data.warning))
        delta = data.qpos[:3] - (start_position if start_position is not None else data.qpos[:3])
        print(json.dumps(dict(status=status, elapsed_simulation_seconds=(step+1)*dt,
                              displacement_m=delta.tolist(), min_clearance_m=min_clearance if np.isfinite(min_clearance) else None,
                              max_tilt_deg=max_tilt, warnings=warnings, send_ms=args.send_ms,
                              terrain=args.terrain, command=list(command), resets=resets,
                              imu_compensation=False, real_servo_interpolation=False), indent=2), flush=True)
        if not args.keyboard and (status in ('fallen', 'nonfinite') or warnings):
            raise RuntimeError(f'Simulation failed: {status}, warnings={warnings}')
    except KeyboardInterrupt:
        pass
    finally:
        if control_window:
            control_window.close()
        if viewer:
            viewer.close()
            time.sleep(1)


if __name__ == '__main__':
    main()
