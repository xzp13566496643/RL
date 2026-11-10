"""Stage 2a确定性策略的CPU MuJoCo实时仿真。"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from hexapod.simulation import physics
from hexapod.simulation.kinematics import LEGS, inverse
from hexapod.simulation.model import (controlled_joint_indices,
                                      controlled_qpos_indices, make_model)
from hexapod.simulation.perturbations import Perturbations

from .policy import ActorPolicy


COMMANDS = {
    'stand': np.array((0., 0., 0.), dtype=np.float32),
    'forward': np.array((1., 0., 0.), dtype=np.float32),
    'backward': np.array((-1., 0., 0.), dtype=np.float32),
    'left': np.array((0., 1., 0.), dtype=np.float32),
    'right': np.array((0., -1., 0.), dtype=np.float32),
}


class Stage2Simulator:
    control_dt = .02
    substeps = 4
    gait_frequency_hz = 1.25
    max_speed_deg_s = 180.

    def __init__(self, policy_path, *, seed=42):
        self.policy = ActorPolicy(policy_path)
        model, template = make_model(
            dynamic=True, terrain='flat', perturbations=Perturbations.sample(),
            imu=True, mjx_compatible=True)
        self.model = model
        self._initial_qpos = template.qpos.copy()
        self.qpos_indices = controlled_qpos_indices(model)
        self.joint_ids = controlled_joint_indices(model)
        self.dof_indices = model.jnt_dofadr[self.joint_ids]
        self.lower = model.jnt_range[self.joint_ids, 0].copy()
        self.upper = model.jnt_range[self.joint_ids, 1].copy()
        self.center = np.concatenate([
            inverse(i, np.asarray(leg.mount) + leg.home)
            for i, leg in enumerate(LEGS)])
        self.center = np.clip(self.center, self.lower, self.upper)
        self.action_scale = np.tile(np.deg2rad((20., 20., 40.)), 6)
        self.max_delta = np.deg2rad(self.max_speed_deg_s) * self.control_dt
        self.body_id = model.body('body').id
        self.body_geom_id = model.geom('body_collision').id
        self.foot_geom_ids = np.array([
            model.geom(f'foot{i}_collision').id for i in range(6)], dtype=int)
        self.rng = np.random.default_rng(seed)
        self.command_name = 'stand'
        self.velocity_command = COMMANDS['stand'].copy()
        self.speed = .15
        self.reset()

    def reset(self):
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:] = self._initial_qpos
        self.data.qpos[self.qpos_indices] = self.center
        self.data.qpos[3:7] = (1., 0., 0., 0.)
        self.data.qvel[:] = 0.
        self.data.ctrl[:] = self.center
        self.command = self.center.copy()
        self.previous_action = np.zeros(18, dtype=np.float32)
        self.phase = 0.
        self.steps = 0
        mujoco.mj_forward(self.model, self.data)

    def set_command(self, name, speed=.15):
        if name not in COMMANDS:
            raise ValueError(f'未知命令：{name}')
        speed = float(speed)
        if not np.isfinite(speed) or not .1 <= speed <= .2:
            raise ValueError('演示速度必须在0.1–0.2 m/s范围内')
        self.command_name = name
        self.speed = speed
        self.velocity_command = COMMANDS[name] * (0. if name == 'stand' else speed)
        if name == 'stand':
            self.phase = 0.

    def _normalize_joint_position(self, position):
        position = np.clip(position, self.lower, self.upper)
        delta = position - self.center
        return np.where(delta < 0., delta / (self.center - self.lower),
                        delta / (self.upper - self.center))

    def observation(self):
        rotation = self.data.xmat[self.body_id].reshape(3, 3)
        gravity = rotation.T @ np.array((0., 0., -1.))
        gravity += .01 * self.rng.normal(size=3)
        gravity /= max(np.linalg.norm(gravity), 1e-12)
        joint_position = self.data.qpos[self.qpos_indices]
        joint_position = joint_position + .01 * self.rng.normal(size=18)
        joint_velocity = self.data.qvel[self.dof_indices] + .05 * self.rng.normal(size=18)
        gyro = self.data.sensordata[3:6] * .25 + .005 * self.rng.normal(size=3)
        observation = np.concatenate((
            gyro, gravity, self.velocity_command * np.array((2., 2., 1.)),
            self._normalize_joint_position(joint_position), joint_velocity / 10.,
            self.previous_action, (np.cos(self.phase), np.sin(self.phase))))
        return np.clip(observation, -100., 100.).astype(np.float32)

    def step(self):
        action = self.policy(self.observation())
        desired = self.center + np.clip(action, -1., 1.) * self.action_scale
        self.command += np.clip(desired - self.command, -self.max_delta, self.max_delta)
        self.data.ctrl[:] = np.clip(self.command, self.lower, self.upper)
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
        if self.command_name == 'stand':
            self.phase = 0.
        else:
            self.phase = (self.phase + 2 * np.pi * self.gait_frequency_hz
                          * self.control_dt) % (2 * np.pi)
        self.previous_action = action
        self.steps += 1
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise RuntimeError('MuJoCo状态出现非有限数值')

    def _contacts(self):
        feet = np.zeros(6, dtype=bool)
        body = False
        for contact in self.data.contact:
            if contact.dist > 0.:
                continue
            pair = (contact.geom1, contact.geom2)
            for robot, other in (pair, pair[::-1]):
                if self.model.geom_group[other] != 2:
                    continue
                body |= robot == self.body_geom_id
                matches = np.flatnonzero(self.foot_geom_ids == robot)
                if len(matches):
                    feet[matches[0]] = True
        return bool(body), feet

    def status(self):
        rotation = self.data.xmat[self.body_id].reshape(3, 3)
        local_velocity = rotation.T @ self.data.qvel[:3]
        matrix = np.empty(9)
        mujoco.mju_quat2Mat(matrix, self.data.qpos[3:7])
        matrix = matrix.reshape(3, 3)
        rpy = np.rad2deg((np.arctan2(matrix[2, 1], matrix[2, 2]),
                         np.arcsin(np.clip(-matrix[2, 0], -1., 1.)),
                         np.arctan2(matrix[1, 0], matrix[0, 0])))
        body_contact, feet = self._contacts()
        return {
            'iteration': self.policy.iteration,
            'total_steps': self.policy.total_steps,
            'command': self.command_name,
            'target_velocity_m_s': self.velocity_command[:2].tolist(),
            'actual_velocity_m_s': local_velocity[:2].tolist(),
            'target_speed_m_s': float(np.linalg.norm(self.velocity_command[:2])),
            'actual_speed_m_s': float(np.linalg.norm(local_velocity[:2])),
            'position_m': self.data.qpos[:3].tolist(),
            'rpy_deg': rpy.tolist(),
            'feet_in_contact': int(np.count_nonzero(feet)),
            'body_contact': body_contact,
            'simulation_time_s': float(self.data.time),
        }
