"""JAX Actor/Critic；层尺寸和tanh高斯策略与PyTorch版本一致。"""
from __future__ import annotations

import jax
import jax.numpy as jnp


ACTION_EPSILON = 1e-6


def _linear(key, input_dim, output_dim, final_std=None):
    if final_std is None:
        limit = jnp.sqrt(6./(input_dim+output_dim))
        weight = jax.random.uniform(key, (input_dim, output_dim), minval=-limit, maxval=limit)
    else:
        weight = final_std*jax.random.normal(key, (input_dim, output_dim))
    return {'weight': weight, 'bias': jnp.zeros(output_dim)}


def init_actor(key, input_dim=65, hidden=(64, 64), output_dim=18, initial_log_std=-1.8):
    keys = jax.random.split(key, len(hidden)+1)
    dims = (input_dim,)+tuple(hidden)+(output_dim,)
    layers = tuple(_linear(keys[i], dims[i], dims[i+1],
                           .01 if i == len(dims)-2 else None)
                   for i in range(len(dims)-1))
    log_std = jnp.asarray(initial_log_std, dtype=jnp.float32)
    if log_std.ndim == 0:
        log_std = jnp.full((output_dim,), log_std)
    elif log_std.shape != (output_dim,):
        raise ValueError(f'initial_log_std须为标量或{output_dim}维，实际{log_std.shape}')
    return {'layers': layers, 'log_std': log_std}


def init_critic(key, input_dim=94, hidden=(256, 128, 64)):
    keys = jax.random.split(key, len(hidden)+1)
    dims = (input_dim,)+tuple(hidden)+(1,)
    return {'layers': tuple(_linear(keys[i], dims[i], dims[i+1])
                            for i in range(len(dims)-1))}


def _mlp(layers, value):
    for layer in layers[:-1]:
        value = jnp.tanh(value@layer['weight']+layer['bias'])
    last = layers[-1]
    return value@last['weight']+last['bias']


def actor_mean(params, observation):
    return _mlp(params['layers'], observation)


def critic_value(params, observation):
    return jnp.squeeze(_mlp(params['layers'], observation), axis=-1)


def _normal_log_prob(raw, mean, std):
    return -.5*((raw-mean)/std)**2-jnp.log(std)-.5*jnp.log(2*jnp.pi)


def action_log_probability(params, observation, action):
    bounded = jnp.clip(action, -1.+ACTION_EPSILON, 1.-ACTION_EPSILON)
    raw = jnp.arctanh(bounded)
    mean = actor_mean(params, observation)
    std = jnp.exp(jnp.clip(params['log_std'], -3., 0.))
    log_jacobian = 2*(jnp.log(2.)-raw-jax.nn.softplus(-2*raw))
    return jnp.sum(_normal_log_prob(raw, mean, std)-log_jacobian, axis=-1)


def sample_action(params, observation, key):
    mean = actor_mean(params, observation)
    std = jnp.exp(jnp.clip(params['log_std'], -3., 0.))
    raw = mean+std*jax.random.normal(key, mean.shape)
    action = jnp.tanh(raw)
    log_jacobian = 2*(jnp.log(2.)-raw-jax.nn.softplus(-2*raw))
    log_probability = jnp.sum(_normal_log_prob(raw, mean, std)-log_jacobian, axis=-1)
    entropy = jnp.sum(.5*jnp.log(2*jnp.pi*jnp.e*std**2))
    return action, log_probability, entropy


def deterministic_action(params, observation):
    return jnp.tanh(actor_mean(params, observation))
