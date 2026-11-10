"""MJX/JAX端到端GPU并行PPO训练入口。"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml

from hexapod.rl.metrics import write_training_plot

from .environment import MJXEnvironment
from .networks import init_actor, init_critic
from .ppo import make_collector, make_updater
from .reward import REWARD_NAMES, RewardConfig


def staggered_episode_steps(key, num_envs, episode_steps):
    """可复现地随机分配近似均匀的初始Episode进度。"""
    if num_envs <= 0 or episode_steps <= 0:
        raise ValueError('环境数和Episode步数必须为正整数')
    evenly_spaced = jnp.floor(
        jnp.arange(num_envs, dtype=jnp.float32)*episode_steps/num_envs
    ).astype(jnp.int32)
    return evenly_spaced[jax.random.permutation(key, num_envs)]


def resolve_velocity_commands(values):
    """解析手动选择的命令预设；不会依据训练轮数自动切换。"""
    selected = str(values.get('selected_preset', 'custom'))
    presets = values.get('presets', {})
    if selected == 'custom':
        resolved = dict(values.get('custom', {}))
    else:
        if selected not in presets:
            raise ValueError(f'未知速度命令预设：{selected}')
        resolved = dict(presets[selected])
    overrides = values.get('overrides', {}) or {}
    resolved.update({key: value for key, value in overrides.items()
                     if value is not None})
    required = ('probabilities', 'axis_speed_range_m_s',
                'omni_speed_range_m_s', 'xy_yaw_speed_range_m_s',
                'yaw_speed_range_rad_s')
    missing = [name for name in required if name not in resolved]
    if missing:
        raise ValueError(f'速度命令配置缺少字段：{missing}')
    probabilities = resolved['probabilities']
    categories = ('stationary', 'axis_xy', 'omni_xy', 'xy_yaw')
    if set(probabilities) != set(categories):
        raise ValueError(f'probabilities必须且只能包含{categories}')
    probability_values = np.asarray([probabilities[name] for name in categories], float)
    if (not np.isfinite(probability_values).all()
            or np.any(probability_values < 0.)
            or not np.isclose(probability_values.sum(), 1., atol=1e-8)):
        raise ValueError('四类速度命令概率必须为非负有限数且总和等于1')
    return selected, resolved


def save_checkpoint(path, payload):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as file:
        pickle.dump(jax.device_get(payload), file)
    temporary.replace(path)


def save_policy_npz(path, params, iteration, total_steps,
                    current_velocity_reward_weight):
    params = jax.device_get(params)
    arrays = {'iteration': np.array(iteration), 'total_steps': np.array(total_steps),
              'velocity_reward_weight': np.array(current_velocity_reward_weight),
              'actor_log_std': np.asarray(params['actor']['log_std'])}
    for index, layer in enumerate(params['actor']['layers']):
        arrays[f'actor_w{index}'] = np.asarray(layer['weight'])
        arrays[f'actor_b{index}'] = np.asarray(layer['bias'])
    for index, layer in enumerate(params['critic']['layers']):
        arrays[f'critic_w{index}'] = np.asarray(layer['weight'])
        arrays[f'critic_b{index}'] = np.asarray(layer['bias'])
    np.savez(path, **arrays)


def load_pytorch_weights(path):
    """读取convert_checkpoint生成的NPZ；优化器按JAX训练重新初始化。"""
    source = np.load(path)
    actor = {'layers': tuple({'weight': jnp.asarray(source[f'actor_w{i}']),
                              'bias': jnp.asarray(source[f'actor_b{i}'])}
                             for i in range(3)),
             'log_std': jnp.asarray(source['actor_log_std'])}
    critic = {'layers': tuple({'weight': jnp.asarray(source[f'critic_w{i}']),
                               'bias': jnp.asarray(source[f'critic_b{i}'])}
                              for i in range(4))}
    return {'actor': actor, 'critic': critic}, int(source['source_iteration']), int(
        source['source_total_steps'])


def train(config_path, *, iterations=None, num_envs=None, init_from=None, resume=None):
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    num_envs = int(num_envs or config['num_envs'])
    iterations = int(iterations or config['total_iterations'])
    rollout_steps = int(config['rollout_steps'])
    episode_steps = int(config['episode_steps'])
    if jax.default_backend() != 'gpu':
        raise RuntimeError(f'MJX训练要求GPU，当前后端为{jax.default_backend()}')

    task_config_path = config_path.parent/config.get('task_config', 'flat_walk.yaml')
    flat_config = yaml.safe_load(task_config_path.read_text(encoding='utf-8'))
    reward_config = RewardConfig(**flat_config['task']['reward'])
    command_stage, command_config = resolve_velocity_commands(
        flat_config['velocity_commands'])
    probabilities = command_config['probabilities']
    scene = flat_config['scene']
    environment = flat_config['environment']
    env = MJXEnvironment(
        control_dt=environment['control_dt'], episode_seconds=flat_config['task']['episode_seconds'],
        gait_frequency_hz=environment['gait_frequency_hz'],
        max_speed_deg_s=environment['max_speed_deg_s'],
        initial_joint_deg=scene['initial_joint_deg'],
        joint_bias_rad=environment['joint_bias_deg']*jnp.pi/180.,
        settle_seconds=scene['settle_seconds'],
        stationary_probability=probabilities['stationary'],
        axis_probability=probabilities['axis_xy'],
        omni_probability=probabilities['omni_xy'],
        xy_yaw_probability=probabilities['xy_yaw'],
        axis_speed_range=command_config['axis_speed_range_m_s'],
        omni_speed_range=command_config['omni_speed_range_m_s'],
        xy_yaw_speed_range=command_config['xy_yaw_speed_range_m_s'],
        yaw_speed_range=command_config['yaw_speed_range_rad_s'],
        command_stage=command_stage,
        healthy_clearance_range_m=flat_config['task']['healthy_clearance_range_m'],
        reward_airborne_threshold_m=reward_config.feet_airborne_threshold_m)

    key = jax.random.PRNGKey(flat_config['seed'])
    key, actor_key, critic_key, reset_key, progress_key = jax.random.split(key, 5)
    network_config = config.get('network', {})
    initial_log_std = np.asarray(
        network_config.get('initial_log_std', -1.8), dtype=np.float32)
    if initial_log_std.ndim == 0:
        initial_log_std = np.full(18, initial_log_std, dtype=np.float32)
    elif initial_log_std.shape == (3,):
        # 18维真实顺序为每腿[髋、大腿、小腿]，不是按关节类型分成三段。
        initial_log_std = np.tile(initial_log_std, 6)
    elif initial_log_std.shape != (18,):
        raise ValueError(
            'network.initial_log_std须为标量、[髋,大腿,小腿]三维或18维')
    if not np.isfinite(initial_log_std).all():
        raise ValueError('network.initial_log_std必须全部为有限数值')
    params = {
        'actor': init_actor(actor_key, initial_log_std=initial_log_std),
        'critic': init_critic(critic_key),
    }
    source_iteration = source_steps = 0
    if init_from is not None:
        params, source_iteration, source_steps = load_pytorch_weights(init_from)
    ppo = config['ppo']
    optimizer = optax.chain(optax.clip_by_global_norm(ppo['max_gradient_norm']),
                            optax.adam(ppo['learning_rate']))
    optimizer_state = optimizer.init(params)
    start_iteration = 0
    total_steps = 0
    if resume is not None:
        if init_from is not None:
            raise ValueError('--resume与--init-from不能同时使用')
        with Path(resume).open('rb') as file:
            checkpoint = pickle.load(file)
        if checkpoint.get('format_version') != 1:
            raise ValueError('不支持的MJX checkpoint格式')
        params = jax.tree_util.tree_map(jnp.asarray, checkpoint['params'])
        optimizer_state = jax.tree_util.tree_map(jnp.asarray, checkpoint['optimizer_state'])
        start_iteration = int(checkpoint['iteration'])
        total_steps = int(checkpoint['total_steps'])
        if iterations < start_iteration:
            raise ValueError('总迭代数不能小于checkpoint已有迭代数')
    collector = make_collector(env, reward_config, rollout_steps,
                               ppo['gamma'], ppo['gae_lambda'])
    updater = make_updater(optimizer, ppo['epochs'], ppo['clip_range'],
                           ppo['value_coefficient'], ppo['entropy_coefficient'],
                           ppo['target_kl'])
    reset = jax.jit(jax.vmap(env.reset))

    keys = jax.random.split(reset_key, num_envs)
    state, actor_obs, critic_obs = reset(keys)
    initial_episode_steps = staggered_episode_steps(
        progress_key, num_envs, episode_steps)
    state = state._replace(steps=initial_episode_steps)
    jax.block_until_ready(actor_obs)
    episode_return = jnp.zeros(num_envs, dtype=jnp.float32)
    # 初始进度只随机化计数，首个Episode并非完整采样，不能作为完整回报。
    episode_return_valid = initial_episode_steps == 0

    run_dir = (Path(resume).resolve().parent if resume is not None else
               Path(config['output_dir'])/f"run-{datetime.now():%Y%m%d-%H%M%S}")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps({
        'command_stage': command_stage,
        'command_probabilities': probabilities,
        'axis_speed_range_m_s': command_config['axis_speed_range_m_s'],
        'omni_speed_range_m_s': command_config['omni_speed_range_m_s'],
        'xy_yaw_speed_range_m_s': command_config['xy_yaw_speed_range_m_s'],
        'yaw_speed_range_rad_s': command_config['yaw_speed_range_rad_s'],
        'resumed': resume is not None,
        'restored_iteration': start_iteration,
        'restored_total_steps': total_steps,
        'initial_episode_step_min': int(jnp.min(initial_episode_steps)),
        'initial_episode_step_max': int(jnp.max(initial_episode_steps)),
    }, ensure_ascii=False), flush=True)
    (run_dir/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    (run_dir/'task-config.json').write_text(
        json.dumps(flat_config, indent=2), encoding='utf-8')
    if init_from is not None:
        (run_dir/'initialization.json').write_text(json.dumps({
            'pytorch_checkpoint': str(Path(init_from).resolve()),
            'source_iteration': source_iteration,
            'source_total_steps': source_steps,
            'optimizer': 'JAX Adam重新初始化',
        }, indent=2, ensure_ascii=False), encoding='utf-8')
    metrics_file = (run_dir/'metrics.jsonl').open('a', encoding='utf-8')
    completed_iteration = start_iteration
    current_velocity_reward_weight = reward_config.linear_velocity_weight
    try:
        for iteration in range(start_iteration+1, iterations+1):
            start = time.perf_counter()
            (state, actor_obs, critic_obs, episode_return,
             episode_return_valid, key, batch) = collector(
                params['actor'], params['critic'], state, actor_obs, critic_obs,
                episode_return, episode_return_valid, key,
                jnp.asarray(current_velocity_reward_weight, dtype=jnp.float32))
            params, optimizer_state, update_metrics, update_active, updates_done = updater(
                params, optimizer_state, batch)
            jax.block_until_ready(update_metrics)
            seconds = time.perf_counter()-start
            samples = num_envs*rollout_steps
            total_steps += samples
            completed_iteration = iteration
            values = jax.device_get(update_metrics)
            valid = batch.valid_mask.astype(jnp.float32)
            valid_count = jnp.maximum(jnp.sum(valid), 1.)
            def valid_mean(value):
                weights = valid.reshape(valid.shape+(1,)*(value.ndim-valid.ndim))
                extra = int(np.prod(value.shape[valid.ndim:])) if value.ndim > valid.ndim else 1
                return jnp.sum(value*weights)/(valid_count*extra)
            record = {
                'iteration': iteration, 'total_steps': total_steps,
                'velocity_reward_weight': current_velocity_reward_weight,
                'samples_per_second': samples/seconds,
                'valid_transition_fraction': float(jnp.mean(valid)),
                'mean_step_reward': float(valid_mean(batch.reward)),
                'estimated_episode_return': float(
                    valid_mean(batch.reward)/environment['control_dt']
                    * flat_config['task']['episode_seconds']),
                'alive_fraction': 1.0,
                'termination_fraction': float(valid_mean(
                    (batch.body_contact | batch.nonfinite).astype(jnp.float32))),
                'body_contact_fraction': float(valid_mean(
                    batch.body_contact.astype(jnp.float32))),
                'timeout_fraction': float(valid_mean(batch.timeout.astype(jnp.float32))),
                'nonfinite_fraction': float(valid_mean(
                    batch.nonfinite.astype(jnp.float32))),
                'reset_count': int(jnp.sum(batch.reset)),
                'reset_timeout_count': int(jnp.sum(batch.timeout)),
                'reset_body_contact_count': int(jnp.sum(batch.body_contact)),
                'reset_nonfinite_count': int(jnp.sum(batch.nonfinite)),
                'mean_abs_sampled_action': float(valid_mean(jnp.abs(batch.action))),
                'total_loss': float(values[0]), 'actor_loss': float(values[1]),
                'critic_loss': float(values[2]), 'entropy': float(values[3]),
                'approximate_kl': float(values[4]), 'clip_fraction': float(values[5]),
                'ppo_updates': int(updates_done),
                'kl_early_stop': not bool(update_active),
                'action_std_mean': float(jnp.exp(jnp.clip(
                    params['actor']['log_std'], -3., 0.)).mean()),
            }
            completed_count = jnp.sum(batch.completed_episode_return_valid)
            completed_return_sum = jnp.sum(jnp.where(
                batch.completed_episode_return_valid,
                batch.completed_episode_return, 0.))
            completed_count, completed_return_sum = jax.device_get(
                (completed_count, completed_return_sum))
            record.update({
                'completed_episode_count': int(completed_count),
                'partial_completed_episode_count': int(
                    jnp.sum(batch.reset)-jnp.sum(
                        batch.completed_episode_return_valid)),
                'mean_completed_episode_return': (
                    float(completed_return_sum/completed_count)
                    if completed_count else None),
            })
            reward_weights = valid[..., None]
            mean_reward_rates = jax.device_get(
                jnp.sum(batch.reward_rates*reward_weights, axis=(0, 1))/valid_count)
            record.update({f'reward_rate_{name}': float(value)
                           for name, value in zip(REWARD_NAMES, mean_reward_rates)})
            # Actor观测6:9是缩放后的速度命令；判断是否为零不受缩放影响。
            moving_mask = jnp.any(batch.actor_observation[..., 6:9] != 0., axis=-1)
            stationary_count = jnp.sum((~moving_mask)*valid)
            moving_count = jnp.sum(moving_mask*valid)
            stationary_reward_rates = jnp.where(
                stationary_count > 0,
                jnp.sum(batch.reward_rates*((~moving_mask)*valid)[..., None], axis=(0, 1))
                / stationary_count,
                jnp.full((len(REWARD_NAMES),), jnp.nan))
            moving_reward_rates = jnp.where(
                moving_count > 0,
                jnp.sum(batch.reward_rates*(moving_mask*valid)[..., None], axis=(0, 1))
                / moving_count,
                jnp.full((len(REWARD_NAMES),), jnp.nan))
            stationary_return = jnp.where(
                stationary_count > 0,
                jnp.sum(jnp.where(~moving_mask, batch.reward*valid, 0.))/stationary_count
                / environment['control_dt']*flat_config['task']['episode_seconds'],
                jnp.nan)
            moving_return = jnp.where(
                moving_count > 0,
                jnp.sum(jnp.where(moving_mask, batch.reward*valid, 0.))/moving_count
                / environment['control_dt']*flat_config['task']['episode_seconds'],
                jnp.nan)
            (stationary_return, moving_return, stationary_reward_rates,
             moving_reward_rates) = jax.device_get(
                (stationary_return, moving_return, stationary_reward_rates,
                 moving_reward_rates))
            record.update({
                'stationary_estimated_return': (float(stationary_return)
                                                if np.isfinite(stationary_return) else None),
                'moving_estimated_return': (float(moving_return)
                                            if np.isfinite(moving_return) else None),
            })
            record.update({
                f'stationary_reward_rate_{name}': (
                    float(value) if np.isfinite(value) else None)
                for name, value in zip(REWARD_NAMES, stationary_reward_rates)})
            record.update({
                f'moving_reward_rate_{name}': (
                    float(value) if np.isfinite(value) else None)
                for name, value in zip(REWARD_NAMES, moving_reward_rates)})
            tracking = batch.tracking_errors
            direction_count = jnp.sum(tracking[..., 3]*valid)
            record.update({
                'tracking_error_linear_speed_m_s': float(valid_mean(tracking[..., 0])),
                'tracking_error_linear_direction_deg': float(jnp.where(
                    direction_count > 0.,
                    jnp.sum(tracking[..., 1]*tracking[..., 3]*valid)/direction_count, 0.)),
                'tracking_error_yaw_velocity_rad_s': float(valid_mean(tracking[..., 2])),
                'feet_height_rmse_m': float(valid_mean(tracking[..., 4])),
                'swing_height_rmse_m': float(valid_mean(tracking[..., 5])),
                'swing_airborne_fraction': float(valid_mean(tracking[..., 6])),
                'stance_contact_fraction': float(valid_mean(tracking[..., 7])),
                'foot_first_contact_events': float(valid_mean(tracking[..., 8])),
                'foot_air_time_mean_s': float(valid_mean(tracking[..., 9])),
                'swing_valid_airborne_fraction': float(valid_mean(tracking[..., 10])),
                'adjacent_same_state_fraction': float(valid_mean(tracking[..., 11])),
                'coordination_valid_fraction': float(valid_mean(tracking[..., 12])),
                'home_l2_coefficient_a': float(valid_mean(tracking[..., 13])),
                'home_l2_coefficient_b': float(valid_mean(tracking[..., 14])),
            })
            metrics_file.write(json.dumps(record)+'\n')
            metrics_file.flush()
            print(json.dumps(record, ensure_ascii=False), flush=True)
            plot_interval = int(config.get('plot_interval', 0))
            if plot_interval > 0 and iteration % plot_interval == 0:
                write_training_plot(run_dir/'metrics.jsonl', run_dir/'training_curves.svg')
            if iteration % int(config['checkpoint_interval']) == 0:
                checkpoint = {
                    'format_version': 1, 'iteration': iteration,
                    'total_steps': total_steps, 'params': params,
                    'optimizer_state': optimizer_state, 'config': config}
                save_checkpoint(run_dir/'latest.pkl', checkpoint)
                save_policy_npz(run_dir/'latest-policy.npz', params, iteration, total_steps,
                                current_velocity_reward_weight)
                # latest文件便于恢复和评估；编号文件保证本机定期同步时不会因
                # 下一次保存覆盖而漏掉某个50轮节点。
                checkpoint_dir = run_dir/'checkpoints'
                checkpoint_dir.mkdir(exist_ok=True)
                stem = f'iteration-{iteration:08d}'
                save_checkpoint(checkpoint_dir/f'{stem}.pkl', checkpoint)
                save_policy_npz(checkpoint_dir/f'{stem}-policy.npz', params, iteration,
                                total_steps, current_velocity_reward_weight)
    except KeyboardInterrupt:
        print(f'收到中断，正在保存第{completed_iteration}轮...', flush=True)
    finally:
        metrics_file.close()
    save_checkpoint(run_dir/'latest.pkl', {
        'format_version': 1, 'iteration': completed_iteration, 'total_steps': total_steps,
        'params': params, 'optimizer_state': optimizer_state, 'config': config})
    save_policy_npz(run_dir/'latest-policy.npz', params, completed_iteration, total_steps,
                    current_velocity_reward_weight)
    if int(config.get('plot_interval', 0)) > 0:
        write_training_plot(run_dir/'metrics.jsonl', run_dir/'training_curves.svg')
    return run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/mjx_flat_walk.yaml')
    parser.add_argument('--iterations', type=int)
    parser.add_argument('--num-envs', type=int)
    parser.add_argument('--init-from', type=Path, help='convert_checkpoint导出的NPZ权重')
    parser.add_argument('--resume', type=Path, help='继续MJX的latest.pkl')
    args = parser.parse_args()
    print('设备:', jax.devices(), flush=True)
    print('训练输出:', train(args.config, iterations=args.iterations,
                              num_envs=args.num_envs, init_from=args.init_from,
                              resume=args.resume), flush=True)


if __name__ == '__main__':
    main()
