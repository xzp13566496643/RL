"""站立物理环境：18维绝对目标角映射、角度限幅、目标变化限速与物理步进。

本阶段不是Gym训练环境；已有IMU观测但没有奖励。state仅供调试，不是网络输入。
"""
import mujoco
import numpy as np

from hexapod_sim.kinematics import LEGS, inverse
from hexapod_sim.model import (make_model, controlled_joint_indices, controlled_qpos,
                               controlled_qpos_indices, set_controlled_qpos)
from hexapod_sim.perturbations import Perturbations
from hexapod_sim import physics
from .imu import IMU, IMUConfig
from .scene import sample_spawn


class StandEnv:
    def __init__(self, control_dt=.02, max_speed_deg_s=60.,
                 joint_bias_deg=3., mount_bias_mm=0., length_bias_mm=0., imu_config=None,
                 scene_config=None, velocity_command=(0., 0., 0.),
                 velocity_command_scale=(2., 2., .25),
                 critic_target_clearance_m=.0792, critic_clearance_scale_m=.020,
                 critic_alignment_scale_deg=30., critic_joint_velocity_scale_rad_s=10.,
                 critic_privileged_clip=10.):
        values = (control_dt, max_speed_deg_s)
        if any(not np.isfinite(v) or v <= 0 for v in values):
            raise ValueError('控制周期和目标角速度限制必须为有限正数')
        self.control_dt = control_dt
        self.max_delta = np.deg2rad(max_speed_deg_s) * control_dt
        self.error_bounds = (joint_bias_deg, mount_bias_mm, length_bias_mm)
        self.imu = IMU(imu_config if imu_config is not None else IMUConfig())
        self.scene_config = scene_config
        self.velocity_command_scale = np.asarray(velocity_command_scale, dtype=np.float32)
        if (self.velocity_command_scale.shape != (3,)
                or not np.isfinite(self.velocity_command_scale).all()
                or np.any(self.velocity_command_scale <= 0)):
            raise ValueError('目标速度观测尺度必须是三个有限正数')
        self.set_velocity_command(velocity_command)
        critic_scales = (critic_clearance_scale_m, critic_alignment_scale_deg,
                         critic_joint_velocity_scale_rad_s, critic_privileged_clip)
        if any(not np.isfinite(v) or v <= 0 for v in critic_scales):
            raise ValueError('Critic特权观测尺度和裁剪值必须为有限正数')
        if not np.isfinite(critic_target_clearance_m):
            raise ValueError('Critic目标净空必须为有限数')
        self.critic_target_clearance_m = critic_target_clearance_m
        self.critic_clearance_scale_m = critic_clearance_scale_m
        self.critic_alignment_scale_rad = np.deg2rad(critic_alignment_scale_deg)
        self.critic_joint_velocity_scale_rad_s = critic_joint_velocity_scale_rad_s
        self.critic_privileged_clip = critic_privileged_clip
        # IK仅在初始化计算固定参考角；step不调用IK或步态。
        self.reference = np.concatenate([
            inverse(i, np.asarray(leg.mount) + leg.home) for i, leg in enumerate(LEGS)])

    def reset(self, seed=None):
        # 不传seed则每局换误差；显式传相同seed可复现实验。
        geometry_seed, imu_seed, scene_seed = np.random.SeedSequence(seed).spawn(3)
        rng = np.random.default_rng(scene_seed)
        self.ground_normal = np.array([0., 0., 1.])
        self.slope_deg = 0.
        initial_offset = np.zeros(18)
        body_quat = np.array([1., 0., 0., 0.])
        self.patch = None
        if self.scene_config is not None:
            cfg = self.scene_config
            self.patch = sample_spawn(cfg, rng)
            yaw = self.patch['yaw_rad']
            body_quat = np.array([np.cos(yaw/2), 0., 0., np.sin(yaw/2)])
            initial_offset = np.deg2rad(rng.uniform(-cfg.initial_joint_deg, cfg.initial_joint_deg, 18))
            # 只随机世界Yaw，不额外施加Roll/Pitch；释放后姿态由接触决定。
        # 显式seed保持与原偏置演示相同的几何误差，方便横向对比。
        self.errors = Perturbations.sample(seed if seed is not None else int(geometry_seed.generate_state(1)[0]),
                                           *self.error_bounds)
        self.platform_enabled = self.scene_config is not None  # 兼容旧调试字段；现在表示启用图集。
        self.model, self.data = make_model(True, terrain='atlas' if self.platform_enabled else 'flat',
                                           perturbations=self.errors, imu=True)
        self.imu_steps = round(self.imu.config.period_s / self.model.opt.timestep)
        if self.imu_steps < 1 or not np.isclose(
                self.imu_steps * self.model.opt.timestep, self.imu.config.period_s):
            raise ValueError('IMU周期必须是物理步长的整数倍')
        self.substeps = round(self.control_dt / self.model.opt.timestep)
        if self.substeps < 1 or not np.isclose(
                self.substeps * self.model.opt.timestep, self.control_dt):
            raise ValueError('控制周期必须是物理步长的整数倍')
        physical = self.model.actuator_ctrlrange.copy()
        bias = self.errors.joint_targets(np.zeros(18))
        # 名义命令本身及施加零位误差后的实际目标均不超执行器范围。
        self.lower = np.maximum(physical[:, 0], physical[:, 0] - bias)
        self.upper = np.minimum(physical[:, 1], physical[:, 1] - bias)
        if np.any(self.lower > self.upper):
            raise ValueError('偏置过大：参考站姿附近不存在合法动作区间')
        self.center = np.clip(self.reference, self.lower, self.upper)
        # 初始动作含独立大扰动；后续step由调用方动作接管，不再次叠加大扰动。
        self.command = np.clip(self.reference + initial_offset, self.lower, self.upper)
        self.initial_joint_offset = self.command - self.center
        actual = self.errors.joint_targets(self.command)
        set_controlled_qpos(self.model, self.data, actual)
        if self.patch is not None:
            self.data.qpos[:2] = self.patch['spawn_xy']
        self.data.qpos[3:7] = body_quat
        self.data.ctrl[:] = actual
        mujoco.mj_forward(self.model, self.data)
        if self.patch is not None:
            # 先放到当地地面上方，再按所有碰撞体的世界Z包围范围精确抬升。
            self.data.qpos[2] += self.ground_height(*self.data.qpos[:2])
            mujoco.mj_forward(self.model, self.data)
        gaps = []
        for gid in range(self.model.ngeom):
            if self.model.geom_bodyid[gid] == 0:
                continue
            size = self.model.geom_size[gid]
            local_n = self.data.geom_xmat[gid].reshape(3, 3).T @ np.array([0., 0., 1.])
            kind = self.model.geom_type[gid]
            if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
                extent = size[0] + size[1] * abs(local_n[2])
            elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
                extent = size[0] * np.linalg.norm(local_n[:2]) + size[1] * abs(local_n[2])
            else:  # 当前模型其余碰撞体只有足球。
                extent = size[0]
            terrain_z = self.ground_height(*self.data.geom_xpos[gid, :2])
            gaps.append(float(self.data.geom_xpos[gid, 2] - extent - terrain_z))
        self.data.qpos[2] += physics.INITIAL_GROUND_CLEARANCE - min(gaps)
        mujoco.mj_forward(self.model, self.data)
        # 障碍块边缘可能与圆形足垫横向相交，而“碰撞体中心射线”看不到该边缘。
        # 只在reset时整体上抬，直到机器人与地形不再初始穿透；不冻结或推进物理。
        for _ in range(12):
            penetration = max((-float(c.dist) for c in self.data.contact
                               if c.dist < 0 and (self.model.geom_group[c.geom1] == 2
                                                  or self.model.geom_group[c.geom2] == 2)),
                              default=0.)
            if penetration <= 1e-6:
                break
            self.data.qpos[2] += penetration + physics.INITIAL_GROUND_CLEARANCE
            mujoco.mj_forward(self.model, self.data)
        ground = self.ground_height(*self.data.qpos[:2])
        self.support_point = np.array([*self.data.qpos[:2], ground])
        self.ground_normal = self.local_ground_normal(*self.data.qpos[:2])
        self.slope_deg = float(np.rad2deg(np.arccos(np.clip(self.ground_normal[2], -1., 1.))))
        # 正式回合前保持本局随机初始目标：先等待形成足端支撑，再额外压实一段时间。
        # 这段物理不采集观测、不计算奖励，也不允许策略接管。
        self.settle_seconds = (self.scene_config.settle_seconds
                               if self.scene_config is not None else 0.)
        initial_xy = self.data.qpos[:2].copy()
        settle_steps = round(self.settle_seconds / self.model.opt.timestep)
        if not np.isclose(settle_steps * self.model.opt.timestep, self.settle_seconds):
            raise ValueError('预稳定时长必须是物理步长的整数倍')
        self.landing_detected = False
        if self.scene_config is not None and self.settle_seconds > 0:
            timeout_steps = int(np.ceil(self.scene_config.landing_timeout_seconds
                                        / self.model.opt.timestep))
            for _ in range(timeout_steps):
                mujoco.mj_step(self.model, self.data)
                if np.count_nonzero(self.terrain_contacts()[1]) >= self.scene_config.landing_min_feet:
                    self.landing_detected = True
                    break
        for _ in range(settle_steps):
            mujoco.mj_step(self.model, self.data)
        self.pre_settle_duration_s = float(self.data.time)
        self.pre_settle_xy_drift_m = float(np.linalg.norm(self.data.qpos[:2] - initial_xy))
        self.pre_settle_foot_contacts = int(np.count_nonzero(self.terrain_contacts()[1]))
        # 预稳定只是构造初始状态：正式回合从静止、t=0和全新IMU历史开始。
        self.data.qvel[:] = 0.
        self.data.qacc[:] = 0.
        self.data.qacc_warmstart[:] = 0.
        self.data.time = 0.
        mujoco.mj_forward(self.model, self.data)
        ground = self.ground_height(*self.data.qpos[:2])
        self.support_point = np.array([*self.data.qpos[:2], ground])
        self.ground_normal = self.local_ground_normal(*self.data.qpos[:2])
        self.physics_steps = 0
        self.previous_action = np.zeros(18, dtype=np.float32)
        self.imu.reset(int(imu_seed.generate_state(1)[0]))
        self.imu.sample(self.data)
        return self.state()

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (18,) or not np.isfinite(action).all():
            raise ValueError('action必须是18维有限数值数组')
        # action=0严格对应HOME；负/正半轴分别映射到本关节安全下限/上限。
        clipped_action = np.clip(action, -1., 1.)
        desired = self.action_to_command(clipped_action)
        self.command += np.clip(desired - self.command, -self.max_delta, self.max_delta)
        self.data.ctrl[:] = self.errors.joint_targets(self.command)
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
            self.physics_steps += 1
            if self.physics_steps % self.imu_steps == 0:
                # mj_step后重新计算当前位置的传感器，避免读取积分前的旧值。
                mujoco.mj_forward(self.model, self.data)
                self.imu.sample(self.data)
        mujoco.mj_forward(self.model, self.data)
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise RuntimeError('物理状态出现非有限数值')
        self.previous_action = clipped_action.astype(np.float32)
        return self.state()

    def action_to_command(self, action):
        """将[-1,1]动作分段映射到安全范围，同时保持零动作对应HOME。"""
        action = np.clip(np.asarray(action, dtype=float), -1., 1.)
        if action.shape != (18,) or not np.isfinite(action).all():
            raise ValueError('action必须是18维有限数值数组')
        return np.where(action < 0., self.center + action * (self.center - self.lower),
                        self.center + action * (self.upper - self.center))

    def command_to_action(self, command):
        """安全范围内关节目标的逆映射，供初始姿态保持和诊断使用。"""
        command = np.clip(np.asarray(command, dtype=float), self.lower, self.upper)
        delta = command - self.center
        negative_span = self.center - self.lower
        positive_span = self.upper - self.center
        return np.where(delta < 0., np.divide(delta, negative_span,
                                               out=np.zeros_like(delta), where=negative_span > 0.),
                        np.divide(delta, positive_span,
                                  out=np.zeros_like(delta), where=positive_span > 0.))

    def observation(self):
        """策略输入：60维归一化IMU历史、18维上一动作、3维目标速度，共81维。"""
        obs = np.concatenate((self.imu.policy_observation(), self.previous_action,
                              self.velocity_command * self.velocity_command_scale))
        return np.clip(obs, -self.imu.config.observation_clip,
                       self.imu.config.observation_clip).astype(np.float32)

    def terrain_contacts(self):
        """返回机身接触、6维足端接触和6维各腿非足端接触状态。"""
        body_geom = self.model.geom('body_collision').id
        feet = {self.model.geom(f'foot{i}_collision').id: i for i in range(6)}
        body_touch = False
        foot_touch = np.zeros(6, dtype=np.float32)
        leg_touch = np.zeros(6, dtype=np.float32)
        for contact in self.data.contact:
            if contact.dist > 0:
                continue
            for robot, other in ((contact.geom1, contact.geom2),
                                 (contact.geom2, contact.geom1)):
                if self.model.geom_group[other] != 2:
                    continue
                body_touch |= robot == body_geom
                if robot in feet:
                    foot_touch[feet[robot]] = 1.
                body_id = self.model.geom_bodyid[robot]
                body_name = self.model.body(body_id).name if body_id else ''
                if body_name.startswith('leg') and len(body_name) > 3 and body_name[3].isdigit():
                    leg_touch[int(body_name[3])] = 1.
        return bool(body_touch), foot_touch, leg_touch

    def critic_observation(self):
        """训练专用149维特权观测；不得作为Actor或实机推理输入。"""
        state = self.state()
        _, foot_touch, leg_touch = self.terrain_contacts()
        joint_ids = controlled_joint_indices(self.model)
        velocity_indices = self.model.jnt_dofadr[joint_ids]
        ranges = self.model.actuator_ctrlrange
        midpoint = ranges.mean(axis=1)
        half_range = (ranges[:, 1] - ranges[:, 0]) / 2.
        joint_position = (controlled_qpos(self.model, self.data) - midpoint) / half_range
        actuator_target = (self.data.ctrl - midpoint) / half_range
        privileged = np.concatenate((
            np.array([np.deg2rad(state['ground_alignment_deg'])
                      / self.critic_alignment_scale_rad,
                      (state['body_clearance_m'] - self.critic_target_clearance_m)
                      / self.critic_clearance_scale_m], dtype=np.float32),
            foot_touch,
            joint_position.astype(np.float32),
            (self.data.qvel[velocity_indices]
             / self.critic_joint_velocity_scale_rad_s).astype(np.float32),
            actuator_target.astype(np.float32),
            leg_touch,
        ))
        privileged = np.clip(privileged, -self.critic_privileged_clip,
                             self.critic_privileged_clip)
        obs = np.concatenate((self.observation(), privileged)).astype(np.float32)
        if obs.shape != (149,) or not np.isfinite(obs).all():
            raise RuntimeError('Critic特权观测维度不匹配或出现非有限值')
        return obs

    def set_velocity_command(self, command):
        """设置(vx, vy, wz)目标；当前站立任务固定为零，接口为后续行走预留。"""
        command = np.asarray(command, dtype=np.float32)
        if command.shape != (3,) or not np.isfinite(command).all():
            raise ValueError('目标速度命令必须是有限的(vx, vy, wz)三维数组')
        self.velocity_command = command.copy()

    def ground_height(self, x, y):
        """从世界上方向下发射射线，查询指定XY处最高的实际地形表面。"""
        height, _ = self.support_surface(x, y)
        return height

    def support_surface(self, x, y):
        geomid = np.array([-1], dtype=np.int32)
        distance = mujoco.mj_ray(self.model, self.data, np.array([x, y, 5.]), np.array([0., 0., -1.]),
                                 np.array([0, 0, 1, 0, 0, 0], dtype=np.uint8), True, -1, geomid)
        if distance < 0:
            raise RuntimeError('机身下方未找到地形')
        return float(5. - distance), int(geomid[0])

    def local_ground_normal(self, x, y):
        """返回机身正下方射线命中表面的真实法向，统一指向世界坐标上方。"""
        _, geomid = self.support_surface(x, y)
        normal = self.data.geom_xmat[geomid].reshape(3, 3)[:, 2].copy()
        if normal[2] < 0:
            normal *= -1
        return normal / np.linalg.norm(normal)

    def state(self):
        rotation = self.data.body('body').xmat.reshape(3, 3)
        ground, geomid = self.support_surface(*self.data.qpos[:2])
        ground_normal = self.data.geom_xmat[geomid].reshape(3, 3)[:, 2].copy()
        if ground_normal[2] < 0:
            ground_normal *= -1
        ground_normal /= np.linalg.norm(ground_normal)
        self.ground_normal = ground_normal
        self.slope_deg = float(np.rad2deg(np.arccos(np.clip(ground_normal[2], -1., 1.))))
        body_axis = rotation[:, 2]
        alignment_rad = float(np.arccos(np.clip(np.dot(body_axis, ground_normal), -1., 1.)))
        # 圆柱机身沿地面法向的支撑半径；据此计算机身外表面而非原点到地面的净空。
        axial = abs(float(np.dot(body_axis, ground_normal)))
        body_extent = (physics.BODY_HALF_HEIGHT * axial
                       + physics.BODY_RADIUS * np.sqrt(max(0., 1. - axial ** 2)))
        ground_point = np.array([self.data.qpos[0], self.data.qpos[1], ground])
        body_clearance = float(np.dot(self.data.qpos[:3] - ground_point, ground_normal)
                               - body_extent)
        on_patch = True
        if self.patch is not None:
            delta = np.abs(self.data.qpos[:2] - self.patch['patch_center'])
            on_patch = bool(np.all(delta < self.patch['patch_half_extent']))
        return dict(time_s=float(self.data.time), height_m=float(self.data.qpos[2]),
                    ground_height_m=ground,
                    clearance_m=float(self.data.qpos[2] - ground),
                    body_clearance_m=body_clearance,
                    ground_alignment_deg=float(np.rad2deg(alignment_rad)),
                    ground_normal=ground_normal.tolist(),
                    on_platform=on_patch,
                    terrain_type='flat' if self.patch is None else self.patch['kind'],
                    terrain_level=0 if self.patch is None else self.patch['level'],
                    slope_deg=float(self.slope_deg),
                    reset_landing_detected=bool(self.landing_detected),
                    reset_settle_duration_s=float(self.pre_settle_duration_s),
                    reset_xy_drift_m=float(self.pre_settle_xy_drift_m),
                    reset_foot_contacts=int(self.pre_settle_foot_contacts),
                    tilt_deg=float(np.rad2deg(np.arccos(np.clip(rotation[2, 2], -1, 1)))),
                    command_deg=np.rad2deg(self.command).tolist(),
                    actuator_target_deg=np.rad2deg(self.data.ctrl).tolist(),
                    actual_joint_deg=np.rad2deg(controlled_qpos(self.model, self.data)).tolist())
