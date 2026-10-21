"""加载训练checkpoint，以确定性Actor独立评估并可视化站立效果。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np
import torch

from .actor import StandActor
from .config import training_config_from_dict
from .gym_env import StandGymEnv
from .train import resolve_device


def load_policy(checkpoint_path, device_name=None):
    """按checkpoint内保存的网络配置重建Actor，只加载部署所需权重。"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if checkpoint.get('format_version') != 1 or 'config' not in checkpoint:
        raise ValueError('不支持的checkpoint格式或缺少训练配置')
    config = training_config_from_dict(checkpoint['config'])
    device = resolve_device(device_name or config.device)
    net = config.network
    actor = StandActor(net.initial_log_std, net.actor_hidden_dims, net.activation).to(device)
    actor.load_state_dict(checkpoint['actor'])
    actor.eval()
    return checkpoint, config, device, actor


def make_environment(config):
    env_cfg = config.environment
    return StandGymEnv(
        task_config=config.task, scene_config=config.scene, imu_config=config.imu,
        control_dt=env_cfg.control_dt, max_speed_deg_s=env_cfg.max_speed_deg_s,
        joint_bias_deg=env_cfg.joint_bias_deg, mount_bias_mm=env_cfg.mount_bias_mm,
        length_bias_mm=env_cfg.length_bias_mm,
        velocity_command=env_cfg.velocity_command,
        velocity_command_scale=env_cfg.velocity_command_scale,
        critic_alignment_scale_deg=env_cfg.critic_alignment_scale_deg,
        critic_joint_velocity_scale_rad_s=env_cfg.critic_joint_velocity_scale_rad_s,
        critic_privileged_clip=env_cfg.critic_privileged_clip)


def evaluate(checkpoint_path, *, episodes=1, seed=1000, device_name=None,
             headless=False, pause=True, zero_action=False, initial_pause_s=1.5):
    checkpoint, config, device, actor = load_policy(checkpoint_path, device_name)
    env = make_environment(config)
    episode_records = []
    print(json.dumps({
        'checkpoint': str(Path(checkpoint_path)),
        'iteration': int(checkpoint['iteration']),
        'training_steps': int(checkpoint['total_steps']),
        'device': str(device),
        'policy': 'zero_action_home_baseline' if zero_action else 'deterministic_tanh_mean',
    }, ensure_ascii=False), flush=True)
    try:
        for episode in range(episodes):
            observation, reset_info = env.reset(seed=seed + episode)
            physics = env.task.env
            viewer = None
            reward_sums = {}
            try:
                if not headless:
                    viewer = mujoco.viewer.launch_passive(physics.model, physics.data)
                    viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.1, 135, -30
                    viewer.cam.lookat[:] = physics.data.qpos[:3]
                    max_offset = float(np.max(np.abs(np.rad2deg(physics.initial_joint_offset))))
                    viewer.set_texts([(
                        mujoco.mjtFontScale.mjFONTSCALE_100,
                        mujoco.mjtGridPos.mjGRID_TOPLEFT,
                        f'INITIAL STATE | max joint offset={max_offset:.1f} deg',
                        f'Physics paused for {initial_pause_s:.1f}s before policy control')])
                    deadline = time.monotonic() + initial_pause_s
                    while viewer.is_running() and time.monotonic() < deadline:
                        viewer.sync()
                        time.sleep(.02)
                # 窗口创建和初始观察时间不计入物理实时同步，否则开窗耗时会让仿真追帧。
                wall_start = time.monotonic()
                while True:
                    tensor = torch.as_tensor(observation, dtype=torch.float32,
                                             device=device).unsqueeze(0)
                    if zero_action:
                        action = np.zeros(18, dtype=np.float32)
                    else:
                        with torch.no_grad():
                            action = actor.deterministic(tensor)[0].cpu().numpy()
                    observation, reward, terminated, truncated, info = env.step(action)
                    for name, value in info.get('reward_rates', {}).items():
                        reward_sums[name] = reward_sums.get(name, 0.) + value * physics.control_dt
                    if viewer:
                        state = info['state']
                        viewer.set_texts([(
                            mujoco.mjtFontScale.mjFONTSCALE_100,
                            mujoco.mjtGridPos.mjGRID_TOPLEFT,
                            f'policy eval | t={state["time_s"]:.2f}s | reward={info["episode_return"]:.2f}',
                            f'{state["terrain_type"]} L{state["terrain_level"]} | '
                            f'align={state["ground_alignment_deg"]:.1f} deg | '
                            f'clearance={state["body_clearance_m"]:.3f} m')])
                        viewer.sync()
                        time.sleep(max(0., wall_start + physics.data.time - time.monotonic()))
                    if terminated or truncated or (viewer and not viewer.is_running()):
                        break
                state = info.get('state', {})
                record = {
                    'episode': episode + 1,
                    'seed': seed + episode,
                    'episode_seed': reset_info['episode_seed'],
                    'return': float(info.get('episode_return', reward)),
                    'duration_s': float(state.get('time_s', physics.data.time)),
                    'result': info.get('reason', 'window_closed'),
                    'terrain_type': state.get('terrain_type'),
                    'terrain_level': state.get('terrain_level'),
                    'ground_alignment_deg': state.get('ground_alignment_deg'),
                    'body_clearance_m': state.get('body_clearance_m'),
                    'reward_components': reward_sums,
                }
                episode_records.append(record)
                print(json.dumps(record, ensure_ascii=False), flush=True)
                if viewer and pause and episode == episodes - 1 and viewer.is_running():
                    viewer.set_texts([(
                        mujoco.mjtFontScale.mjFONTSCALE_100,
                        mujoco.mjtGridPos.mjGRID_TOPLEFT,
                        f'EVALUATION {record["result"]} | return={record["return"]:.2f}',
                        'Paused for inspection; close window or Ctrl+C to exit')])
                    print('评估结束，画面已暂停；关闭窗口或按Ctrl+C退出。', flush=True)
                    while viewer.is_running():
                        viewer.sync()
                        time.sleep(.02)
            finally:
                if viewer:
                    viewer.close()
        returns = np.asarray([record['return'] for record in episode_records])
        summary = {
            'episodes': len(episode_records),
            'mean_return': float(returns.mean()) if len(returns) else None,
            'std_return': float(returns.std()) if len(returns) else None,
            'body_contact_rate': (sum(record['result'] == 'body_contact'
                                      for record in episode_records) / len(episode_records)
                                  if episode_records else None),
        }
        print('汇总：' + json.dumps(summary, ensure_ascii=False), flush=True)
        return episode_records, summary
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'))
    parser.add_argument('--headless', action='store_true', help='不打开窗口，适合批量统计')
    parser.add_argument('--zero-action', action='store_true',
                        help='不用Actor，运行action=0（HOME）基线以作同种子对照')
    parser.add_argument('--initial-pause', type=float, default=1.5,
                        help='有窗口时先冻结物理若干秒，以观察随机初始姿态')
    parser.add_argument('--no-pause', action='store_true', help='最后一局结束后直接关闭窗口')
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error('--episodes必须为正整数')
    if not np.isfinite(args.initial_pause) or args.initial_pause < 0:
        parser.error('--initial-pause必须为有限非负数')
    try:
        evaluate(args.checkpoint, episodes=args.episodes, seed=args.seed,
                 device_name=args.device, headless=args.headless,
                 pause=not args.no_pause, zero_action=args.zero_action,
                 initial_pause_s=args.initial_pause)
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
