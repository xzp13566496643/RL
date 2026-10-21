"""端到端站立PPO持续训练入口：日志、checkpoint及断点恢复。"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime
import json
import os
from pathlib import Path
import random

import numpy as np
import torch

from .actor import StandActor
from .collector import SingleEnvCollector
from .config import DEFAULT_CONFIG, TrainingConfig
from .critic import StandCritic
from .gym_env import StandGymEnv
from .plot_metrics import write_training_plot
from .ppo import PPOUpdater
from .rollout import RolloutBuffer


def resolve_device(name: str) -> torch.device:
    if name == 'auto':
        return (torch.device('cuda', torch.cuda.current_device())
                if torch.cuda.is_available() else torch.device('cpu'))
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('配置要求CUDA，但当前PyTorch检测不到CUDA')
    if name == 'cuda':
        return torch.device('cuda', torch.cuda.current_device())
    return torch.device(name)


def make_components(config: TrainingConfig):
    if config.rollout.num_envs != 1:
        raise ValueError('当前train.py只支持num_envs=1；并行采集器尚未接入')
    device = resolve_device(config.device)
    env_cfg = config.environment
    env = StandGymEnv(
        task_config=config.task, scene_config=config.scene, imu_config=config.imu,
        control_dt=env_cfg.control_dt, max_speed_deg_s=env_cfg.max_speed_deg_s,
        joint_bias_deg=env_cfg.joint_bias_deg, mount_bias_mm=env_cfg.mount_bias_mm,
        length_bias_mm=env_cfg.length_bias_mm,
        velocity_command=env_cfg.velocity_command,
        velocity_command_scale=env_cfg.velocity_command_scale,
        critic_alignment_scale_deg=env_cfg.critic_alignment_scale_deg,
        critic_joint_velocity_scale_rad_s=env_cfg.critic_joint_velocity_scale_rad_s,
        critic_privileged_clip=env_cfg.critic_privileged_clip)
    net = config.network
    actor = StandActor(net.initial_log_std, net.actor_hidden_dims, net.activation).to(device)
    critic = StandCritic(net.critic_hidden_dims, net.activation).to(device)
    updater = PPOUpdater(actor, critic, config.ppo)
    buffer = RolloutBuffer(config.rollout.num_steps, config.rollout.num_envs, device)
    return device, env, actor, critic, updater, buffer


def checkpoint_payload(iteration, total_steps, actor, critic, updater, config):
    payload = {
        'format_version': 1,
        'iteration': int(iteration),
        'total_steps': int(total_steps),
        'actor': actor.state_dict(),
        'critic': critic.state_dict(),
        'optimizer': updater.optimizer.state_dict(),
        'config': asdict(config),
        'python_random_state': random.getstate(),
        'numpy_random_state': np.random.get_state(),
        'torch_random_state': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload['torch_cuda_random_state'] = torch.cuda.get_rng_state_all()
    return payload


def save_checkpoint(path, iteration, total_steps, actor, critic, updater, config):
    """先写同目录临时文件再替换，避免中断留下半个checkpoint。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(checkpoint_payload(iteration, total_steps, actor, critic, updater, config), temporary)
    os.replace(temporary, path)


def load_checkpoint(path, actor, critic, updater, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get('format_version') != 1:
        raise ValueError('不支持的checkpoint格式')
    actor.load_state_dict(checkpoint['actor'])
    critic.load_state_dict(checkpoint['critic'])
    updater.optimizer.load_state_dict(checkpoint['optimizer'])
    random.setstate(checkpoint['python_random_state'])
    np.random.set_state(checkpoint['numpy_random_state'])
    torch.set_rng_state(checkpoint['torch_random_state'].cpu())
    if device.type == 'cuda' and 'torch_cuda_random_state' in checkpoint:
        torch.cuda.set_rng_state_all([state.cpu()
                                      for state in checkpoint['torch_cuda_random_state']])
    return int(checkpoint['iteration']), int(checkpoint['total_steps'])


def explained_variance(values, returns):
    variance = torch.var(returns, unbiased=False)
    if variance <= 1e-12:
        return float('nan')
    return float(1. - torch.var(returns - values, unbiased=False) / variance)


def train(config=DEFAULT_CONFIG, *, resume=None, output_dir=None):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device, env, actor, critic, updater, buffer = make_components(config)
    start_iteration = total_steps = 0
    if resume is not None:
        start_iteration, total_steps = load_checkpoint(resume, actor, critic, updater, device)
        if config.total_iterations < start_iteration:
            raise ValueError('总迭代数不能小于checkpoint已有迭代数')
        run_dir = Path(output_dir) if output_dir else Path(resume).resolve().parent
    else:
        stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        run_dir = Path(output_dir) if output_dir else Path(config.checkpoint_dir) / f'run-{stamp}'
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'config.json').write_text(json.dumps(asdict(config), indent=2), encoding='utf-8')
    metrics_path = run_dir / 'metrics.jsonl'
    collector = SingleEnvCollector(env, actor, critic, device=device,
                                   seed=config.seed + start_iteration)
    try:
        with metrics_path.open('a', encoding='utf-8') as metrics_file:
            for iteration in range(start_iteration + 1, config.total_iterations + 1):
                collection = collector.collect(
                    buffer, gamma=config.rollout.gamma,
                    gae_lambda=config.rollout.gae_lambda,
                    normalize_advantages=config.rollout.normalize_advantages)
                old_values = buffer.old_values.detach().clone()
                returns = buffer.returns.detach().clone()
                mean_reward = float(buffer.rewards.mean())
                ppo = updater.update(buffer)
                total_steps += config.rollout.num_steps * config.rollout.num_envs
                std = actor.log_std.detach().clamp(-3., 0.).exp()
                variance_score = explained_variance(old_values, returns)
                record = {
                    'iteration': iteration, 'total_steps': total_steps,
                    'mean_step_reward': mean_reward,
                    'estimated_episode_return': (mean_reward / config.environment.control_dt
                                                 * config.task.episode_seconds),
                    'completed_episodes': collection.completed_episodes,
                    'terminated_episodes': collection.terminated_episodes,
                    'truncated_episodes': collection.truncated_episodes,
                    'termination_fraction': (collection.terminated_episodes
                                             / collection.completed_episodes
                                             if collection.completed_episodes else None),
                    'mean_abs_sampled_action': collection.mean_abs_sampled_action,
                    'mean_abs_policy_action': collection.mean_abs_policy_action,
                    'mean_completed_return': (float(np.mean(collection.episode_returns))
                                              if collection.episode_returns else None),
                    'actor_loss': ppo.actor_loss, 'critic_loss': ppo.critic_loss,
                    'entropy': ppo.entropy, 'approximate_kl': ppo.approximate_kl,
                    'clip_fraction': ppo.clip_fraction,
                    'gradient_norm_before_clip': ppo.gradient_norm,
                    'ppo_updates': ppo.updates, 'ppo_epochs': ppo.epochs_completed,
                    'kl_early_stop': ppo.early_stopped,
                    'explained_variance_before_update': (variance_score
                                                         if np.isfinite(variance_score) else None),
                    'action_std_mean': float(std.mean()),
                    'action_std_min': float(std.min()), 'action_std_max': float(std.max()),
                }
                record.update({f'reward_rate_{name}': value
                               for name, value in collection.mean_reward_rates.items()})
                metrics_file.write(json.dumps(record, allow_nan=False) + '\n')
                metrics_file.flush()
                if iteration % config.log_interval == 0:
                    print(json.dumps(record, ensure_ascii=False), flush=True)
                if iteration % config.checkpoint_interval == 0:
                    save_checkpoint(run_dir / 'latest.pt', iteration, total_steps,
                                    actor, critic, updater, config)
                    save_checkpoint(run_dir / f'iteration-{iteration:06d}.pt', iteration,
                                    total_steps, actor, critic, updater, config)
                if iteration % config.plot_interval == 0:
                    write_training_plot(metrics_path, run_dir / 'training_curves.svg')
                buffer.clear()
        save_checkpoint(run_dir / 'latest.pt', config.total_iterations, total_steps,
                        actor, critic, updater, config)
        write_training_plot(metrics_path, run_dir / 'training_curves.svg')
    finally:
        env.close()
    return run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iterations', type=int, help='覆盖config中的总迭代数')
    parser.add_argument('--rollout-steps', type=int, help='覆盖每轮单环境采样步数')
    parser.add_argument('--mini-batch-size', type=int, help='覆盖PPO mini-batch')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'))
    parser.add_argument('--seed', type=int)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    config = DEFAULT_CONFIG
    if args.iterations is not None:
        config = replace(config, total_iterations=args.iterations)
    if args.rollout_steps is not None:
        config = replace(config, rollout=replace(config.rollout, num_steps=args.rollout_steps))
    if args.mini_batch_size is not None:
        config = replace(config, ppo=replace(config.ppo, mini_batch_size=args.mini_batch_size))
    if args.device is not None:
        config = replace(config, device=args.device)
    if args.seed is not None:
        config = replace(config, seed=args.seed)
    run_dir = train(config, resume=args.resume, output_dir=args.output_dir)
    print(f'训练输出：{run_dir}', flush=True)


if __name__ == '__main__':
    main()
