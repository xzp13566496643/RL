"""与CPU评估一致的V3纯JAX奖励。"""
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class RewardConfig:
    linear_velocity_weight: float = 4.
    linear_velocity_sigma: float = .05
    stationary_velocity_weight: float = 4.
    yaw_velocity_weight: float = 2.
    yaw_velocity_sigma: float = .05
    stationary_yaw_weight: float = 2.
    stationary_yaw_deadzone_rad_s: float = .05
    stationary_yaw_quadratic_weight: float = 10.
    velocity_command_threshold: float = .05
    body_height_weight: float = 1.
    body_height_falloff_m: float = .0533
    body_tilt_weight: float = 1.
    body_tilt_free_deg: float = 10.
    body_tilt_full_deg: float = 30.
    gait_swing_height_m: float = .03
    feet_phase_weight: float = 1.2
    feet_phase_no_lift_weight: float = .6
    feet_phase_error_scale_m: float = .015
    feet_airborne_threshold_m: float = .0005
    adjacent_coordination_weight: float = .6
    feet_air_time_weight: float = 22.5
    feet_air_time_threshold_s: float = .1
    feet_air_time_cap_s: float = .3
    feet_height_weight: float = 5.
    feet_height_target_m: float = .03
    feet_height_min_peak_m: float = .0005
    phase_boundary_window_s: float = .02
    gait_frequency_hz: float = 1.25
    home_l2_weight: float = 1.5
    # 仅用于兼容旧V3配置；当前HOME L2不再按命令状态分段。
    home_l2_stationary_weight: float = 1.5
    home_l2_moving_weight: float = .3
    home_l1_stationary_weight: float = .1
    action_rate_weight: float = .3
    action_magnitude_weight: float = .08
    joint_acceleration_weight: float = 1e-5
    torque_weight: float = .001
    roll_pitch_rate_weight: float = .1
    vertical_velocity_weight: float = .2
    feet_slip_weight: float = 10.


FOOT_COUNT_SCALE = 2. / 6.
TRIPOD_A = jnp.array([0, 2, 4])
TRIPOD_B = jnp.array([1, 3, 5])
ADJACENT_LEG_PAIRS = jnp.array(((0, 1), (1, 2), (2, 3),
                                (3, 4), (4, 5), (5, 0)))
REWARD_NAMES = (
    'linear_velocity', 'yaw_velocity', 'body_height', 'body_tilt',
    'feet_phase', 'feet_phase_no_lift', 'adjacent_coordination',
    'feet_air_time', 'feet_height',
    'home_l2', 'home_l1_stationary', 'action_rate', 'action_magnitude',
    'joint_acceleration', 'torque', 'roll_pitch_rate', 'vertical_velocity',
    'feet_slip')


def tripod_swing_mask(phase):
    return jnp.where(
        jnp.mod(phase, 2*jnp.pi) < jnp.pi,
        jnp.array([True, False, True, False, True, False]),
        jnp.array([False, True, False, True, False, True]))


def _phase_boundary(phase, config):
    phase = jnp.mod(phase, 2*jnp.pi)
    width = 2*jnp.pi*config.gait_frequency_hz*config.phase_boundary_window_s
    distance_zero = jnp.minimum(phase, 2*jnp.pi-phase)
    return jnp.minimum(distance_zero, jnp.abs(phase-jnp.pi)) <= width+1e-12


def gait_targets(phase, config):
    """返回摆动掩码、摆动进度和六足目标足垫底部净空。"""
    swing = tripod_swing_mask(phase)
    progress = jnp.mod(phase, jnp.pi)/jnp.pi
    target_height = config.gait_swing_height_m*jnp.sin(jnp.pi*progress)
    return swing, progress, jnp.where(swing, target_height, 0.)


def gait_reward_components(phase, contacts, foot_bottom_clearance, is_moving,
                           previous_air_time, first_contacts,
                           landing_peak_clearance, config):
    """只评价摆动足高度，并结算首次落地的腾空时间和峰值。"""
    swing, _, targets = gait_targets(phase, config)
    logical_contact = contacts | (
        foot_bottom_clearance <= config.feet_airborne_threshold_m)
    valid_airborne = ~logical_contact
    per_foot_phase = jnp.where(
        swing & valid_airborne,
        jnp.exp(-(foot_bottom_clearance-targets)**2
                / config.feet_phase_error_scale_m**2), 0.)
    feet_phase = config.feet_phase_weight*jnp.sum(per_foot_phase)/3.
    feet_phase = jnp.where(is_moving, feet_phase, 0.)
    # 0.5 mm只判定是否有效离地；抬升误差始终使用相对真实地面的净空。
    effective_height = jnp.maximum(foot_bottom_clearance, 0.)
    no_lift_error = (jnp.maximum(targets-effective_height, 0.)
                     / config.gait_swing_height_m)
    feet_phase_no_lift = (-config.feet_phase_no_lift_weight
                          * jnp.sum(jnp.where(swing, no_lift_error**2, 0.))/3.)
    feet_phase_no_lift = jnp.where(is_moving, feet_phase_no_lift, 0.)
    feet_phase = feet_phase+feet_phase_no_lift
    eligible_air = jnp.maximum(
        jnp.minimum(previous_air_time, config.feet_air_time_cap_s)
        - config.feet_air_time_threshold_s, 0.)
    feet_air = config.feet_air_time_weight*FOOT_COUNT_SCALE*jnp.sum(
        eligible_air*first_contacts.astype(jnp.float32))
    feet_air = jnp.where(is_moving, feet_air, 0.)
    peak_error = landing_peak_clearance/config.feet_height_target_m-1.
    valid_height_landing = first_contacts & (
        landing_peak_clearance >= config.feet_height_min_peak_m)
    feet_height = config.feet_height_weight*FOOT_COUNT_SCALE*jnp.sum(
        (1.-peak_error**2)*valid_height_landing.astype(jnp.float32))
    feet_height = jnp.where(is_moving, feet_height, 0.)
    return (feet_phase, feet_phase_no_lift, feet_air, feet_height, swing,
            targets)


def home_l2_coefficients(phase, is_moving, config):
    """兼容旧日志：A/B组始终返回同一个固定HOME L2系数。"""
    del phase, is_moving
    return (jnp.asarray(config.home_l2_weight),
            jnp.asarray(config.home_l2_weight))


def adjacent_coordination_components(phase, contacts, foot_bottom_clearance,
                                     is_moving, config):
    """返回相邻腿同状态占比、有效门控和协调性惩罚率。"""
    # 运动分支沿用原高度阈值状态；静止分支严格使用MuJoCo物理接触。
    airborne = foot_bottom_clearance > config.feet_airborne_threshold_m
    same = airborne[ADJACENT_LEG_PAIRS[:, 0]] == airborne[ADJACENT_LEG_PAIRS[:, 1]]
    same_fraction = jnp.mean(same.astype(jnp.float32))
    valid = is_moving & ~_phase_boundary(phase, config)
    moving_penalty = jnp.where(
        valid, -config.adjacent_coordination_weight*same_fraction, 0.)
    stationary_penalty = (-config.adjacent_coordination_weight
                          * jnp.mean((~contacts).astype(jnp.float32)))
    penalty = jnp.where(is_moving, moving_penalty, stationary_penalty)
    return penalty, same_fraction, valid, airborne


def reward_terms(env, previous_state, state, action, config: RewardConfig,
                 linear_velocity_weight=None):
    """返回奖励率字典；调用方最后乘20ms控制周期。"""
    data = state.data
    rotation = data.xmat[env.body_id]
    angular = data.qvel[3:6]
    local_linear = rotation.T@data.qvel[:3]
    command = previous_state.velocity_command
    xy_command_norm = jnp.linalg.norm(command[:2])
    xy_moving = xy_command_norm > config.velocity_command_threshold
    yaw_moving = jnp.abs(command[2]) > config.velocity_command_threshold
    is_moving = xy_moving | yaw_moving

    linear_error2 = jnp.sum((local_linear[:2]-command[:2])**2)
    yaw_error2 = (angular[2]-command[2])**2

    body_axis = rotation[:, 2]
    axial = jnp.abs(body_axis[2])
    extent = env.body_half_height*axial + env.body_radius*jnp.sqrt(
        jnp.maximum(0., 1.-axial**2))
    clearance = data.qpos[2]-extent
    clearance_error = jnp.maximum(
        jnp.maximum(env.healthy_clearance_min_m-clearance, 0.),
        jnp.maximum(clearance-env.healthy_clearance_max_m, 0.))
    body_height = -config.body_height_weight*jnp.clip(
        clearance_error/config.body_height_falloff_m, 0., 1.)
    angle_deg = jnp.rad2deg(jnp.arccos(jnp.clip(body_axis[2], -1., 1.)))
    body_tilt = -config.body_tilt_weight*jnp.clip(
        (angle_deg-config.body_tilt_free_deg)
        / (config.body_tilt_full_deg-config.body_tilt_free_deg), 0., 1.)

    # 解析圆柱足垫的最低碰撞点相对平地z=0的有符号净空；倾斜时包含边缘
    # 相对底面中心的下降量，且绝不以contact=False代替高度测量。
    foot_bottom_clearance = env._foot_bottom_clearance(data)
    (feet_phase, feet_phase_no_lift, feet_air_time, feet_height, _, _
     ) = gait_reward_components(
        previous_state.phase, state.foot_contacts, foot_bottom_clearance,
        is_moving, state.previous_air_time, state.first_contacts,
        state.landing_peak_clearance, config)
    adjacent_coordination, _, _, _ = adjacent_coordination_components(
        previous_state.phase, state.foot_contacts, foot_bottom_clearance,
        is_moving, config)
    qpos = data.qpos[env.qpos_indices]
    home = qpos-env.center
    weighted_home_error = (env.home_reward_weights*home**2).reshape((6, 3)).sum(axis=1)
    coefficient_a, coefficient_b = home_l2_coefficients(
        previous_state.phase, is_moving, config)
    home_l2 = -(coefficient_a*jnp.sum(weighted_home_error[TRIPOD_A])
                + coefficient_b*jnp.sum(weighted_home_error[TRIPOD_B]))
    home_l1 = jnp.where(
        is_moving, 0., -config.home_l1_stationary_weight*jnp.sum(jnp.abs(home)))
    action = jnp.clip(action, -1., 1.)
    horizontal_world_speed2 = jnp.sum(state.foot_velocity[:, :2]**2, axis=1)
    slip_contact = state.foot_contacts | (
        foot_bottom_clearance <= config.feet_airborne_threshold_m)

    linear_weight = (config.linear_velocity_weight if linear_velocity_weight is None
                     else linear_velocity_weight)
    linear_reward = linear_weight*(
        jnp.exp(-linear_error2/config.linear_velocity_sigma)
        - jnp.exp(-jnp.sum(command[:2]**2)/config.linear_velocity_sigma))
    yaw_reward = config.yaw_velocity_weight*(
        jnp.exp(-yaw_error2/config.yaw_velocity_sigma)
        - jnp.exp(-command[2]**2/config.yaw_velocity_sigma))
    return {
        'linear_velocity': linear_reward,
        'yaw_velocity': yaw_reward,
        'body_height': body_height,
        'body_tilt': body_tilt,
        'feet_phase': feet_phase,
        # 仅用于拆分日志；总Feet Phase已经包含此项，累计总奖励时须排除。
        'feet_phase_no_lift': feet_phase_no_lift,
        'adjacent_coordination': adjacent_coordination,
        'feet_air_time': feet_air_time,
        'feet_height': feet_height,
        'home_l2': home_l2,
        'home_l1_stationary': home_l1,
        'action_rate': (-config.action_rate_weight*jnp.sum(
            (action-previous_state.previous_action)**2)),
        'action_magnitude': (-config.action_magnitude_weight*jnp.sum(action**2)),
        'joint_acceleration': (-config.joint_acceleration_weight
                               * jnp.sum(state.joint_acceleration**2)),
        'torque': (-config.torque_weight
                   * jnp.sum(state.mean_actuator_torque_squared)),
        'roll_pitch_rate': -config.roll_pitch_rate_weight*jnp.sum(angular[:2]**2),
        'vertical_velocity': -config.vertical_velocity_weight*data.qvel[2]**2,
        'feet_slip': -config.feet_slip_weight*FOOT_COUNT_SCALE*jnp.sum(
            horizontal_world_speed2*slip_contact),
    }


def total_reward(env, previous_state, state, action, config: RewardConfig,
                 linear_velocity_weight=None):
    terms = reward_terms(env, previous_state, state, action, config,
                         linear_velocity_weight)
    # feet_phase_no_lift已包含在feet_phase中，不能重复计入总奖励。
    accumulated = sum(value for name, value in terms.items()
                      if name != 'feet_phase_no_lift')
    return accumulated*env.control_dt, terms
