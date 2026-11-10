"""纯JAX批量Rollout、GAE和PPO更新。"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from .networks import (action_log_probability, critic_value, deterministic_action,
                       sample_action)
from .reward import (adjacent_coordination_components, gait_targets,
                     home_l2_coefficients, total_reward)


class Batch(NamedTuple):
    actor_observation: jax.Array
    critic_observation: jax.Array
    action: jax.Array
    old_log_probability: jax.Array
    old_value: jax.Array
    reward: jax.Array
    reward_rates: jax.Array
    tracking_errors: jax.Array
    done: jax.Array
    body_contact: jax.Array
    timeout: jax.Array
    nonfinite: jax.Array
    reset: jax.Array
    bootstrap_value: jax.Array
    completed_episode_return: jax.Array
    completed_episode_return_valid: jax.Array
    valid_mask: jax.Array
    advantage: jax.Array
    returns: jax.Array


def generalized_advantage(reward, value, done, terminal, next_value, valid,
                          gamma=.995, gae_lambda=.95):
    """使用Reset前next_value计算GAE，并在任意Episode边界切断递推。"""
    def gae_step(advantage, inputs):
        reward_t, value_t, done_t, terminal_t, next_value_t, valid_t = inputs
        trace_mask = 1.-done_t.astype(jnp.float32)
        # 不使用 next_value * 0：若真正终止状态本身含 NaN，NaN*0 仍会
        # 污染GAE。真正终止明确替换为零，正常timeout则保留Bootstrap。
        safe_next_value = jnp.where(terminal_t, 0., next_value_t)
        delta = reward_t+gamma*safe_next_value-value_t
        candidate = delta+gamma*gae_lambda*trace_mask*advantage
        advantage = jnp.where(valid_t, candidate, 0.)
        return advantage, advantage

    _, advantage = jax.lax.scan(
        gae_step, jnp.zeros_like(value[-1]),
        (reward, value, done, terminal, next_value, valid), reverse=True)
    return advantage


def batch_reset_finished(env, state, actor_obs, critic_obs, reset_keys, reset_mask):
    """批量生成Reset候选，并仅替换本控制步结束Episode的环境。"""
    def choose(reset_value, current_value):
        mask = reset_mask.reshape(
            reset_mask.shape+(1,)*(reset_value.ndim-reset_mask.ndim))
        return jnp.where(mask, reset_value, current_value)

    # Reset已不包含物理步进。所有候选只做批量正向运动学，随后由mask选择，
    # 让整个rollout scan保持单一GPU批处理图，避免逐环境map和设备条件分支。
    reset_state, reset_actor, reset_critic = jax.vmap(env.reset)(reset_keys)
    return jax.tree_util.tree_map(
        choose, (reset_state, reset_actor, reset_critic),
        (state, actor_obs, critic_obs))


def make_collector(env, reward_config, rollout_steps, gamma=.995, gae_lambda=.95):
    physics = jax.vmap(env.step_physics)
    observe = jax.vmap(env.observations)
    termination_flags = jax.vmap(env.termination_flags)
    def reward_vector(old, new, policy_action, linear_velocity_weight):
        reward, terms = total_reward(
            env, old, new, policy_action, reward_config, linear_velocity_weight)
        return reward, jnp.stack(tuple(terms.values()))
    rewards = jax.vmap(reward_vector, in_axes=(0, 0, 0, None))

    def tracking_vector(old, new):
        rotation = new.data.xmat[env.body_id]
        local_linear = rotation.T @ new.data.qvel[:3]
        target_xy = old.velocity_command[:2]
        actual_xy = local_linear[:2]
        target_speed = jnp.linalg.norm(target_xy)
        actual_speed = jnp.linalg.norm(actual_xy)
        direction_valid = target_speed > 0.
        cosine = jnp.dot(actual_xy, target_xy) / jnp.maximum(
            actual_speed*target_speed, 1e-12)
        direction_deg = jnp.where(
            actual_speed > 1e-8,
            jnp.rad2deg(jnp.arccos(jnp.clip(cosine, -1., 1.))), 180.)
        yaw_error = jnp.abs(new.data.qvel[5]-old.velocity_command[2])
        swing, _, targets = gait_targets(old.phase, reward_config)
        heights = env._foot_bottom_clearance(new.data)
        height_rmse = jnp.sqrt(jnp.mean((heights-targets)**2))
        swing_rmse = jnp.sqrt(jnp.mean(jnp.where(
            swing, (heights-targets)**2, 0.))*2.)
        airborne_fraction = jnp.mean((~new.foot_contacts & swing).astype(jnp.float32))
        stance_fraction = jnp.sum((new.foot_contacts & ~swing).astype(jnp.float32))/3.
        first_events = jnp.sum(new.first_contacts.astype(jnp.float32))
        mean_air = jnp.mean(new.foot_air_time)
        _, adjacent_same, coordination_valid, reward_airborne = (
            adjacent_coordination_components(
                old.phase, new.foot_contacts, heights,
                jnp.any(old.velocity_command != 0.), reward_config))
        swing_valid_airborne = jnp.sum(
            (swing & reward_airborne).astype(jnp.float32))/3.
        is_moving = jnp.any(old.velocity_command != 0.)
        home_coefficient_a, home_coefficient_b = home_l2_coefficients(
            old.phase, is_moving, reward_config)
        return jnp.array((jnp.abs(actual_speed-target_speed), direction_deg,
                          yaw_error, direction_valid.astype(jnp.float32),
                          height_rmse, swing_rmse, airborne_fraction,
                          stance_fraction, first_events, mean_air,
                          swing_valid_airborne, adjacent_same,
                          coordination_valid.astype(jnp.float32),
                          home_coefficient_a, home_coefficient_b))
    tracking = jax.vmap(tracking_vector)

    def collect(actor_params, critic_params, state, actor_obs, critic_obs,
                episode_return, episode_return_valid, key, linear_velocity_weight):
        def step(carry, _):
            (state, actor_obs, critic_obs, episode_return,
             episode_return_valid, key) = carry
            key, action_key, reset_key = jax.random.split(key, 3)
            action_keys = jax.random.split(action_key, actor_obs.shape[0])
            action, logp, _ = jax.vmap(sample_action, in_axes=(None, 0, 0))(
                actor_params, actor_obs, action_keys)
            policy_action = jax.vmap(
                deterministic_action, in_axes=(None, 0))(actor_params, actor_obs)
            value = critic_value(critic_params, critic_obs)
            candidate = physics(state, action)
            reward, reward_rates = rewards(
                state, candidate, policy_action, linear_velocity_weight)
            tracking_errors = tracking(state, candidate)
            body_contact, timeout, nonfinite = termination_flags(candidate)
            just_done = body_contact | timeout | nonfinite
            done = just_done
            valid = jnp.ones_like(just_done, dtype=bool)

            # 先观测Reset前的物理后继状态。正常超时从这里Bootstrap；真正终止
            # 不Bootstrap。随后才替换为新Episode的reset状态，防止价值泄漏。
            candidate_actor, candidate_critic, candidate = observe(candidate)
            bootstrap_value = critic_value(critic_params, candidate_critic)
            reset_keys = jax.random.split(reset_key, actor_obs.shape[0])
            next_state, next_actor, next_critic = batch_reset_finished(
                env, candidate, candidate_actor, candidate_critic,
                reset_keys, just_done)

            accumulated_return = episode_return+reward
            completed_return = jnp.where(just_done, accumulated_return, 0.)
            completed_valid = just_done & episode_return_valid
            next_episode_return = jnp.where(just_done, 0., accumulated_return)
            # 初始随机进度产生的第一个Episode是截短样本，不作为完整回报；
            # 任何独立reset之后的新Episode均从第0帧开始，回报可正常统计。
            next_episode_return_valid = jnp.where(
                just_done, True, episode_return_valid)
            transition = (actor_obs, critic_obs, action, logp, value, reward,
                          reward_rates, tracking_errors, done,
                          body_contact, timeout, nonfinite, just_done,
                          bootstrap_value, completed_return, completed_valid, valid)
            return ((next_state, next_actor, next_critic, next_episode_return,
                     next_episode_return_valid, key), transition)

        carry, values = jax.lax.scan(
            step, (state, actor_obs, critic_obs, episode_return,
                   episode_return_valid, key), None, length=rollout_steps)
        (state, next_actor, next_critic, episode_return,
         episode_return_valid, key) = carry
        (actor_obs, critic_obs, action, logp, value, reward, reward_rates,
         tracking_errors, done, body_contact, timeout, nonfinite, reset,
         bootstrap_value, completed_return, completed_valid, valid) = values

        # next_value始终来自Reset前的物理后继状态：正常超时允许Bootstrap，
        # 真正终止不Bootstrap；两者均切断跨Episode的GAE递推。
        advantage = generalized_advantage(
            reward, value, done, body_contact | nonfinite,
            bootstrap_value, valid, gamma, gae_lambda)
        returns = advantage+value
        batch = Batch(actor_obs, critic_obs, action, logp, value, reward, reward_rates,
                      tracking_errors, done, body_contact, timeout, nonfinite, reset,
                      bootstrap_value, completed_return, completed_valid, valid,
                      advantage, returns)
        return (state, next_actor, next_critic, episode_return,
                episode_return_valid, key, batch)
    return jax.jit(collect)


def flatten_batch(batch):
    return jax.tree_util.tree_map(lambda x: x.reshape((-1,)+x.shape[2:]), batch)


def make_updater(optimizer, epochs=3, clip_range=.2, value_coefficient=.5,
                 entropy_coefficient=.003, target_kl=.01):
    def loss_fn(params, batch):
        actor, critic = params['actor'], params['critic']
        logp = action_log_probability(actor, batch.actor_observation, batch.action)
        value = critic_value(critic, batch.critic_observation)
        valid = batch.valid_mask.astype(jnp.float32)
        count = jnp.maximum(jnp.sum(valid), 1.)
        def valid_mean(value):
            return jnp.sum(value*valid)/count
        advantage_mean = valid_mean(batch.advantage)
        advantage_variance = valid_mean((batch.advantage-advantage_mean)**2)
        advantage = (batch.advantage-advantage_mean)/jnp.sqrt(
            advantage_variance+1e-8)
        ratio = jnp.exp(logp-batch.old_log_probability)
        surrogate = jnp.minimum(ratio*advantage,
                                jnp.clip(ratio, 1.-clip_range, 1.+clip_range)*advantage)
        actor_loss = -valid_mean(surrogate)
        critic_loss = .5*valid_mean((value-batch.returns)**2)
        std = jnp.exp(jnp.clip(actor['log_std'], -3., 0.))
        entropy = jnp.sum(.5*jnp.log(2*jnp.pi*jnp.e*std**2))
        has_valid = jnp.sum(valid) > 0
        effective_entropy = jnp.where(has_valid, entropy, 0.)
        total = (actor_loss+value_coefficient*critic_loss
                 - entropy_coefficient*effective_entropy)
        log_ratio = logp-batch.old_log_probability
        approximate_kl = valid_mean((ratio-1.)-log_ratio)
        clip_fraction = valid_mean((jnp.abs(ratio-1.) > clip_range).astype(jnp.float32))
        metrics = jnp.array((total, actor_loss, critic_loss, entropy,
                             approximate_kl, clip_fraction, count))
        return total, metrics

    def update(params, opt_state, batch):
        batch = flatten_batch(batch)
        def epoch(_, carry):
            params, opt_state, _, active, updates_done = carry
            (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, batch)
            old_opt_state = opt_state
            updates, candidate_opt_state = optimizer.update(grads, opt_state, params)
            candidate = optax.apply_updates(params, updates)
            # 与CPU版一致：当前策略已超过目标KL时，不再提交本epoch更新。
            accept = active & (metrics[4] <= target_kl) & (metrics[6] > 0)
            params = jax.tree_util.tree_map(
                lambda new, old: jnp.where(accept, new, old), candidate, params)
            opt_state = jax.tree_util.tree_map(
                lambda new, old: jnp.where(accept, new, old),
                candidate_opt_state, old_opt_state)
            return params, opt_state, metrics, accept, updates_done+accept.astype(jnp.int32)
        return jax.lax.fori_loop(0, epochs, epoch,
                                 (params, opt_state, jnp.zeros(7), jnp.array(True),
                                  jnp.array(0, dtype=jnp.int32)))
    return jax.jit(update)
