"""任务基线：固定参考角或人工正弦动作，输出每局奖励与结束原因，不训练。"""
import argparse
import json

import numpy as np

from hexapod_sim.perturbations import add_arguments, from_arguments
from .env import StandEnv
from .task import StandTask, TaskConfig
from . import scene


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes', type=int, default=3)
    p.add_argument('--seconds', type=float, default=10.)
    p.add_argument('--motion', choices=('hold', 'sine'), default='hold')
    add_arguments(p)
    p.set_defaults(joint_bias_deg=3.)
    scene.add_arguments(p)
    args = p.parse_args()
    from_arguments(args, p)
    if args.episodes < 1:
        p.error('episodes必须大于0')
    try:
        task = StandTask(StandEnv(joint_bias_deg=args.joint_bias_deg, mount_bias_mm=args.mount_bias_mm,
                                  length_bias_mm=args.length_bias_mm, scene_config=scene.from_arguments(args)),
                         TaskConfig(episode_seconds=args.seconds))
    except ValueError as exc:
        p.error(str(exc))
    for episode in range(args.episodes):
        seed = args.seed + episode
        obs, initial = task.reset(seed)
        terminated = truncated = False
        max_tilt = 0.
        min_height = float('inf')
        while not (terminated or truncated):
            action = np.zeros(18)
            if args.motion == 'sine':
                action[1] = .3 * np.sin(2 * np.pi * task.env.data.time / 4)
            obs, reward, terminated, truncated, info = task.step(action)
            if 'state' in info:
                max_tilt = max(max_tilt, info['state']['tilt_deg'])
                min_height = min(min_height, info['state']['clearance_m'])
        print(json.dumps(dict(episode=episode, seed=seed, reason=info['reason'],
                              return_total=info['episode_return'], max_tilt_deg=max_tilt,
                              min_clearance_m=min_height if np.isfinite(min_height) else None,
                              slope_deg=initial['state']['slope_deg'],
                              initial_tilt_deg=initial['state']['tilt_deg'],
                              observation_shape=list(obs.shape), reward_rates=info.get('reward_rates')), indent=2))


if __name__ == '__main__':
    main()
