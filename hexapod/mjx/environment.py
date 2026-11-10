"""纯JAX批量六足环境核心：平地reset、动作执行和65/94维观测。"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np

from hexapod.simulation import physics
from hexapod.simulation.kinematics import LEGS, inverse
from hexapod.simulation.model import (controlled_joint_indices,
                                      controlled_joint_home_weights,
                                      controlled_qpos_indices, make_model)
from hexapod.simulation.perturbations import Perturbations


class State(NamedTuple):
    data: mjx.Data
    command: jax.Array
    previous_action: jax.Array
    previous_joint_velocity: jax.Array
    joint_acceleration: jax.Array
    mean_actuator_torque_squared: jax.Array
    velocity_command: jax.Array
    phase: jax.Array
    steps: jax.Array
    foot_air_time: jax.Array
    previous_feet: jax.Array
    foot_peak_clearance: jax.Array
    foot_velocity: jax.Array
    foot_contacts: jax.Array
    previous_air_time: jax.Array
    first_contacts: jax.Array
    landing_peak_clearance: jax.Array
    servo_bias: jax.Array
    key: jax.Array


class MJXEnvironment:
    """单环境JAX函数；外层用vmap得到128个独立环境。"""

    actor_observation_dim = 65
    critic_observation_dim = 94
    action_dim = 18

    def __init__(self, *, control_dt=.02, episode_seconds=10., gait_frequency_hz=1.25,
                 max_speed_deg_s=180., initial_joint_deg=3., joint_bias_rad=.05,
                 settle_seconds=1., stationary_probability=1.,
                 axis_probability=0., omni_probability=0.,
                 xy_yaw_probability=0., axis_speed_range=(.2, .5),
                 omni_speed_range=(.2, .5), xy_yaw_speed_range=(.2, .5),
                 yaw_speed_range=(.1, .3), command_stage='custom',
                 healthy_clearance_range_m=(.0533, .1067),
                 reward_airborne_threshold_m=.0005):
        self.control_dt = float(control_dt)
        self.episode_steps = round(episode_seconds / control_dt)
        self.gait_delta = 2 * np.pi * gait_frequency_hz * control_dt
        self.max_delta = np.deg2rad(max_speed_deg_s) * control_dt
        self.initial_joint_rad = np.deg2rad(initial_joint_deg)
        self.joint_bias_rad = float(joint_bias_rad)
        self.stationary_probability = float(stationary_probability)
        self.axis_probability = float(axis_probability)
        self.omni_probability = float(omni_probability)
        self.xy_yaw_probability = float(xy_yaw_probability)
        self.command_stage = str(command_stage)
        probabilities = np.asarray((self.stationary_probability,
                                    self.axis_probability,
                                    self.omni_probability,
                                    self.xy_yaw_probability))
        if (not np.isfinite(probabilities).all() or np.any(probabilities < 0.)
                or not np.isclose(probabilities.sum(), 1., atol=1e-8)):
            raise ValueError('四类速度命令概率必须为非负有限数且总和等于1')
        def speed_range(name, values):
            values = np.asarray(values, dtype=float)
            if (values.shape != (2,) or not np.isfinite(values).all()
                    or values[0] < 0. or values[0] > values[1]):
                raise ValueError(f'{name}必须是递增的非负有限范围')
            return float(values[0]), float(values[1])
        self.axis_speed_min, self.axis_speed_max = speed_range(
            'axis_speed_range', axis_speed_range)
        self.omni_speed_min, self.omni_speed_max = speed_range(
            'omni_speed_range', omni_speed_range)
        self.xy_yaw_speed_min, self.xy_yaw_speed_max = speed_range(
            'xy_yaw_speed_range', xy_yaw_speed_range)
        self.yaw_speed_min, self.yaw_speed_max = speed_range(
            'yaw_speed_range', yaw_speed_range)
        self.healthy_clearance_min_m = float(healthy_clearance_range_m[0])
        self.healthy_clearance_max_m = float(healthy_clearance_range_m[1])
        self.reward_airborne_threshold_m = float(reward_airborne_threshold_m)
        self.body_half_height = float(physics.BODY_HALF_HEIGHT)
        self.body_radius = float(physics.BODY_RADIUS)
        self.substeps = round(control_dt / physics.TIMESTEP)
        # 保留参数仅兼容旧调用；V3 Reset不再执行任何PD预稳定物理子步。
        del settle_seconds

        cpu_model, cpu_data = make_model(
            dynamic=True, terrain='flat', perturbations=Perturbations.sample(),
            imu=True, mjx_compatible=True)
        self.cpu_model = cpu_model
        self.model = mjx.put_model(cpu_model)
        self.template = mjx.put_data(cpu_model, cpu_data)

        joint_ids = controlled_joint_indices(cpu_model)
        self.home_reward_weights = jnp.asarray(controlled_joint_home_weights(cpu_model))
        self.qpos_indices = jnp.asarray(controlled_qpos_indices(cpu_model))
        self.dof_indices = jnp.asarray(cpu_model.jnt_dofadr[joint_ids])
        ranges = cpu_model.jnt_range[joint_ids]
        self.lower = jnp.asarray(ranges[:, 0])
        self.upper = jnp.asarray(ranges[:, 1])
        reference = np.concatenate([
            inverse(i, np.asarray(leg.mount) + leg.home) for i, leg in enumerate(LEGS)])
        self.center = jnp.asarray(np.clip(reference, ranges[:, 0], ranges[:, 1]))
        # Actor动作按[髋、大腿、小腿]围绕HOME作对称残差映射。
        # 这里按真实18关节顺序逐腿展开，不能依赖连续的关节类型分组。
        action_scale = np.tile(np.deg2rad((20., 20., 40.)), 6)
        if (np.any(np.asarray(self.center)-action_scale < ranges[:, 0]-1e-12)
                or np.any(np.asarray(self.center)+action_scale > ranges[:, 1]+1e-12)):
            raise ValueError('HOME对称动作范围超出MuJoCo实际几何关节限位')
        # 最坏±0.05rad舵机偏置也应无需物理裁剪；若以后改限位则立即报错。
        if (np.any(np.asarray(self.center)-action_scale-self.joint_bias_rad
                   < ranges[:, 0]-1e-12)
                or np.any(np.asarray(self.center)+action_scale+self.joint_bias_rad
                          > ranges[:, 1]+1e-12)):
            raise ValueError('HOME对称动作范围叠加舵机零位偏置后超出几何限位')
        self.action_scale = jnp.asarray(action_scale)
        self.midpoint = jnp.asarray(ranges.mean(axis=1))
        self.half_range = jnp.asarray((ranges[:, 1] - ranges[:, 0]) / 2)
        self.body_id = cpu_model.body('body').id
        self.body_geom_id = cpu_model.geom('body_collision').id
        self.foot_geom_ids = jnp.asarray([
            cpu_model.geom(f'foot{i}_collision').id for i in range(6)])
        foot_sizes = cpu_model.geom_size[np.asarray(self.foot_geom_ids)]
        self.foot_pad_radius = jnp.asarray(foot_sizes[:, 0])
        self.foot_pad_half_height = jnp.asarray(foot_sizes[:, 1])
        self.foot_site_ids = jnp.asarray([
            cpu_model.site(f'foot{i}_contact').id for i in range(6)])
        self.force_limit = float(physics.MAX_TORQUE)

    def action_to_command(self, action):
        action = jnp.clip(action, -1., 1.)
        return self.center + action*self.action_scale

    def command_to_action(self, command):
        """按旧安全范围分段归一化关节位置，仅用于Actor观测9:27。"""
        delta = command-self.center
        return jnp.where(delta < 0., delta/(self.center-self.lower),
                         delta/(self.upper-self.center))

    def _command_sample(self, key):
        (mode_key, axis_key, axis_speed_key, omni_speed_key, omni_angle_key,
         combined_speed_key, combined_angle_key, yaw_speed_key,
         yaw_sign_key) = jax.random.split(key, 9)
        mode = jax.random.uniform(mode_key)
        axis_index = jax.random.randint(axis_key, (), 0, 4)
        axis_speed = jax.random.uniform(
            axis_speed_key, minval=self.axis_speed_min, maxval=self.axis_speed_max)
        axis_commands = jnp.array(((1., 0., 0.), (-1., 0., 0.),
                                   (0., 1., 0.), (0., -1., 0.)))
        axis_command = axis_commands[axis_index]*axis_speed
        omni_speed = jax.random.uniform(
            omni_speed_key, minval=self.omni_speed_min, maxval=self.omni_speed_max)
        omni_angle = jax.random.uniform(omni_angle_key, minval=0., maxval=2*jnp.pi)
        omni_command = jnp.array((omni_speed*jnp.cos(omni_angle),
                                  omni_speed*jnp.sin(omni_angle), 0.))
        combined_speed = jax.random.uniform(
            combined_speed_key, minval=self.xy_yaw_speed_min,
            maxval=self.xy_yaw_speed_max)
        combined_angle = jax.random.uniform(
            combined_angle_key, minval=0., maxval=2*jnp.pi)
        yaw_speed = jax.random.uniform(
            yaw_speed_key, minval=self.yaw_speed_min, maxval=self.yaw_speed_max)
        yaw_sign = jnp.where(jax.random.bernoulli(yaw_sign_key), 1., -1.)
        combined_command = jnp.array((combined_speed*jnp.cos(combined_angle),
                                      combined_speed*jnp.sin(combined_angle),
                                      yaw_sign*yaw_speed))
        axis_end = self.stationary_probability+self.axis_probability
        omni_end = axis_end+self.omni_probability
        return jnp.where(
            mode < self.stationary_probability, jnp.zeros(3),
            jnp.where(mode < axis_end, axis_command,
                      jnp.where(mode < omni_end, omni_command, combined_command)))

    def _feet(self, data):
        return data.site_xpos[self.foot_site_ids]

    def _foot_bottom_clearance(self, data):
        """解析圆柱足垫最低碰撞点到平地z=0的真实竖直净空。"""
        rotation = data.geom_xmat[self.foot_geom_ids]
        local_z_world_z = rotation[:, 2, 2]
        extent = (self.foot_pad_radius*jnp.sqrt(jnp.maximum(
            0., 1.-local_z_world_z**2))
            + self.foot_pad_half_height*jnp.abs(local_z_world_z))
        return data.geom_xpos[self.foot_geom_ids, 2]-extent

    def _contacts(self, data):
        contact = data._impl.contact
        active = contact.dist <= 0.
        pair = contact.geom
        terrain_first = pair[:, 0] == 0
        robot_geom = jnp.where(terrain_first, pair[:, 1], pair[:, 0])
        has_floor = (pair[:, 0] == 0) | (pair[:, 1] == 0)
        valid = active & has_floor
        feet = jnp.any(valid[:, None] &
                       (robot_geom[:, None] == self.foot_geom_ids[None, :]), axis=0)
        body = jnp.any(valid & (robot_geom == self.body_geom_id))
        return body, feet

    def reset(self, key):
        key, command_key, offset_key, bias_key, yaw_key = jax.random.split(key, 5)
        velocity_command = self._command_sample(command_key)
        offset = jax.random.uniform(offset_key, (18,), minval=-self.initial_joint_rad,
                                    maxval=self.initial_joint_rad)
        servo_bias = jax.random.uniform(bias_key, (18,), minval=-self.joint_bias_rad,
                                       maxval=self.joint_bias_rad)
        command = jnp.clip(self.center+offset, self.lower, self.upper)
        actual = jnp.clip(command+servo_bias, self.lower, self.upper)
        qpos = self.template.qpos.at[self.qpos_indices].set(actual)
        yaw = jax.random.uniform(yaw_key, minval=-jnp.pi, maxval=jnp.pi)
        qpos = qpos.at[3:7].set(jnp.array([jnp.cos(yaw/2), 0., 0., jnp.sin(yaw/2)]))
        data = self.template.replace(qpos=qpos, ctrl=actual,
                                     qvel=jnp.zeros_like(self.template.qvel),
                                     qacc=jnp.zeros_like(self.template.qacc),
                                     qacc_warmstart=jnp.zeros_like(
                                         self.template.qacc_warmstart),
                                     time=jnp.array(0.))
        data = mjx.forward(self.model, data)
        feet = self._feet(data)
        foot_clearance = self._foot_bottom_clearance(data)
        _, foot_contacts = self._contacts(data)
        joint_velocity = data.qvel[self.dof_indices]
        state = State(data, command, jnp.zeros(18), joint_velocity,
                      jnp.zeros(18), jnp.zeros(18),
                      velocity_command, jnp.array(0.), jnp.array(0, dtype=jnp.int32),
                      jnp.zeros(6), feet,
                      jnp.zeros(6), jnp.zeros((6, 3)),
                      foot_contacts, jnp.zeros(6),
                      jnp.zeros(6, dtype=bool), jnp.zeros(6), servo_bias, key)
        actor, critic, state = self.observations(state)
        return state, actor, critic

    def step_physics(self, state, action):
        clipped_action = jnp.clip(action, -1., 1.)
        desired = self.action_to_command(clipped_action)
        command = state.command + jnp.clip(desired-state.command,
                                           -self.max_delta, self.max_delta)
        ctrl = jnp.clip(command+state.servo_bias, self.lower, self.upper)
        data = state.data.replace(ctrl=ctrl)
        def substep(_, carry):
            substep_data, torque_squared_sum = carry
            substep_data = mjx.step(self.model, substep_data)
            torque_squared_sum = torque_squared_sum+substep_data.actuator_force**2
            return substep_data, torque_squared_sum
        data, torque_squared_sum = jax.lax.fori_loop(
            0, self.substeps, substep, (data, jnp.zeros(self.action_dim)))
        mean_actuator_torque_squared = torque_squared_sum/self.substeps
        joint_velocity = data.qvel[self.dof_indices]
        joint_acceleration = (
            joint_velocity-state.previous_joint_velocity)/self.control_dt
        phase = jnp.where(jnp.all(state.velocity_command == 0.), 0.,
                          jnp.mod(state.phase+self.gait_delta, 2*jnp.pi))
        _, contacts = self._contacts(data)
        feet = self._feet(data)
        foot_clearance = self._foot_bottom_clearance(data)
        # 奖励事件使用0.5 mm过滤后的逻辑离地状态；物理接触仍原样保留给
        # Critic和Feet Slip，不改变MJX接触求解。
        logical_contact = contacts | (
            foot_clearance <= self.reward_airborne_threshold_m)
        reward_airborne = ~logical_contact
        previous_air = state.foot_air_time
        first_contacts = (~reward_airborne) & (previous_air > 0.)
        air = jnp.where(reward_airborne, previous_air+self.control_dt, 0.)
        # 静止命令不保留运动阶段的腾空历史，防止切换/Reset后错误结算。
        is_moving = jnp.any(state.velocity_command != 0.)
        first_contacts = first_contacts & is_moving
        previous_air = jnp.where(is_moving, previous_air, jnp.zeros(6))
        air = jnp.where(is_moving, air, jnp.zeros(6))
        foot_velocity = (feet-state.previous_feet)/self.control_dt
        # 峰值只在逻辑腾空期间更新，数值始终是真实地面以上的净空。
        peak = jnp.where(
            reward_airborne,
            jnp.maximum(state.foot_peak_clearance,
                        jnp.maximum(foot_clearance, 0.)),
            state.foot_peak_clearance)
        landing_peak = peak
        peak = jnp.where(first_contacts, 0., peak)
        return state._replace(data=data, command=command,
                              previous_action=clipped_action,
                              previous_joint_velocity=joint_velocity,
                              joint_acceleration=joint_acceleration,
                              mean_actuator_torque_squared=(
                                  mean_actuator_torque_squared),
                              phase=phase,
                              steps=state.steps+1, foot_air_time=air,
                              previous_feet=feet, foot_peak_clearance=peak,
                              foot_velocity=foot_velocity, foot_contacts=contacts,
                              previous_air_time=previous_air,
                              first_contacts=first_contacts,
                              landing_peak_clearance=landing_peak)

    def observations(self, state):
        """生成Actor/Critic观测，并返回推进随机数后的状态。"""
        key, gravity_key, qpos_key, qvel_key, linear_key, gyro_key = jax.random.split(
            state.key, 6)
        data = state.data
        rotation = data.xmat[self.body_id]
        gravity = rotation.T @ jnp.array([0., 0., -1.])
        gravity += .01 * jax.random.normal(gravity_key, (3,))
        gravity /= jnp.maximum(jnp.linalg.norm(gravity), 1e-12)
        joint_position = data.qpos[self.qpos_indices]
        # 观测真实机械关节角；本局固定零位偏置不从读数中抵消。
        noisy_position = joint_position + .01 * jax.random.normal(qpos_key, (18,))
        normalized_position = self.command_to_action(jnp.clip(noisy_position,
                                                               self.lower, self.upper))
        joint_velocity = data.qvel[self.dof_indices]
        noisy_joint_velocity = joint_velocity + .05 * jax.random.normal(qvel_key, (18,))
        actor = jnp.concatenate((
            data.sensordata[3:6] * .25 + .005 * jax.random.normal(gyro_key, (3,)),
            gravity, state.velocity_command*jnp.array([2., 2., 1.]),
            normalized_position, noisy_joint_velocity/10., state.previous_action,
            jnp.array([jnp.cos(state.phase), jnp.sin(state.phase)])))
        actor = jnp.clip(actor, -100., 100.)

        # free joint前三维是机身原点世界线速度，后三维是机身坐标角速度。
        local_linear = rotation.T @ data.qvel[:3]
        foot_velocity = state.foot_velocity
        contacts = state.foot_contacts
        body_axis = rotation[:, 2]
        axial = jnp.abs(body_axis[2])
        extent = physics.BODY_HALF_HEIGHT*axial + physics.BODY_RADIUS*jnp.sqrt(
            jnp.maximum(0., 1.-axial**2))
        clearance = data.qpos[2]-extent
        alignment = jnp.arccos(jnp.clip(body_axis[2], -1., 1.))
        privileged = jnp.concatenate((
            local_linear, contacts.astype(jnp.float32),
            jnp.array((alignment, clearance)), foot_velocity.reshape(-1)))
        critic = jnp.concatenate((actor, jnp.clip(privileged, -10., 10.)))
        return actor.astype(jnp.float32), critic.astype(jnp.float32), state._replace(key=key)

    def done(self, state):
        body_contact, timeout, nonfinite = self.termination_flags(state)
        return body_contact | timeout | nonfinite

    def termination_flags(self, state):
        """分别返回机身碰地、正常超时和非有限状态。"""
        body_contact, _ = self._contacts(state.data)
        timeout = state.steps >= self.episode_steps
        nonfinite = ~(jnp.all(jnp.isfinite(state.data.qpos)) &
                      jnp.all(jnp.isfinite(state.data.qvel)))
        return body_contact, timeout, nonfinite

