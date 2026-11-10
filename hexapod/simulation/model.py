"""模型构建：根据几何和物理参数生成单个六足机器人MJCF。"""
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .kinematics import LEGS, COXA, FEMUR, TIBIA, inverse, frames
from . import physics
from .terrain import add_terrain
from .perturbations import Perturbations
from .joint_limits import geometric_limits


def _add_foot_pad_mesh(root, radial_segments=24, height_segments=4):
    """加入平底、球面拱顶的封闭足垫网格；局部原点仍位于球头中心。"""
    radius = physics.FOOT_PAD_RADIUS
    bottom = -physics.FOOT_BALL_RADIUS
    height = physics.FOOT_PAD_DOME_HEIGHT
    sphere_radius = (radius**2 + height**2) / (2 * height)
    sphere_center_z = bottom + height - sphere_radius
    vertices = []
    # 从平底外圈向拱顶生成球面圆环。
    for ring in range(height_segments):
        ring_radius = radius * (height_segments - ring) / height_segments
        z = sphere_center_z + np.sqrt(sphere_radius**2 - ring_radius**2)
        for segment in range(radial_segments):
            angle = 2 * np.pi * segment / radial_segments
            vertices.append((ring_radius * np.cos(angle), ring_radius * np.sin(angle), z))
    apex = len(vertices)
    vertices.append((0., 0., bottom + height))
    base_center = len(vertices)
    vertices.append((0., 0., bottom))

    faces = []
    for ring in range(height_segments - 1):
        lower = ring * radial_segments
        upper = (ring + 1) * radial_segments
        for segment in range(radial_segments):
            nxt = (segment + 1) % radial_segments
            faces.extend(((lower + segment, lower + nxt, upper + nxt),
                          (lower + segment, upper + nxt, upper + segment)))
    last = (height_segments - 1) * radial_segments
    for segment in range(radial_segments):
        nxt = (segment + 1) % radial_segments
        faces.append((last + segment, last + nxt, apex))
        # 底面三角形法向朝下，形成真实的水平支撑面。
        faces.append((base_center, nxt, segment))

    asset = root.find('asset')
    if asset is None:
        asset = ET.SubElement(root, 'asset')
    ET.SubElement(asset, 'mesh', name='foot_pad_dome',
                  vertex=' '.join(f'{value:.9g}' for vertex in vertices for value in vertex),
                  face=' '.join(str(value) for face in faces for value in face))

def controlled_qpos_indices(model):
    """返回18个主动关节在qpos中的地址；被动脚垫球关节不会混入。"""
    return np.array([model.jnt_qposadr[model.joint(f'leg{i}_{part}_joint').id]
                     for i in range(6) for part in ('coxa', 'femur', 'tibia')], dtype=int)


def controlled_joint_indices(model):
    return np.array([model.joint(f'leg{i}_{part}_joint').id
                     for i in range(6) for part in ('coxa', 'femur', 'tibia')], dtype=int)


def controlled_joint_home_weights(model):
    """按模型中的实际主动关节名称返回HOME L2权重。"""
    by_part = {'coxa': 1., 'femur': .5, 'tibia': .25}
    result = []
    for joint_id in controlled_joint_indices(model):
        name = model.joint(int(joint_id)).name
        matches = [weight for part, weight in by_part.items()
                   if name.endswith(f'_{part}_joint')]
        if len(matches) != 1:
            raise ValueError(f'无法由主动关节名称确定HOME权重：{name}')
        result.append(matches[0])
    return np.asarray(result, dtype=float)


def controlled_joint_ranges(model):
    return model.jnt_range[controlled_joint_indices(model)].copy()


def controlled_qpos(model, data):
    return data.qpos[controlled_qpos_indices(model)].copy()


def set_controlled_qpos(model, data, values):
    values = np.asarray(values)
    if values.shape != (18,):
        raise ValueError('主动关节位置必须为18维')
    data.qpos[controlled_qpos_indices(model)] = values


def make_model(dynamic=False, terrain='flat', slope_deg=5.0, obstacle_mm=8.0, perturbations=None, imu=False,
               ground_quat=None, mjx_compatible=False):
    """创建机器人模型。

    ``mjx_compatible``只供GPU并行后端使用：MJX目前不支持模型中所有潜在的
    cylinder-mesh碰撞组合，因此将拱形Mesh足垫替换为同半径、同底面高度的
    解析圆柱足垫。默认值为False，现有CPU训练和可视化模型完全不变。
    """
    errors = perturbations if perturbations is not None else Perturbations.sample()
    root = ET.Element('mujoco', model='Hexapod geometry preview')
    ET.SubElement(root, 'compiler', angle='radian')
    ET.SubElement(root, 'option', timestep=str(physics.TIMESTEP), integrator='implicitfast',
                  gravity=' '.join(map(str, physics.GRAVITY)) if dynamic else '0 0 0',
                  impratio=str(physics.CONTACT_IMPRATIO),
                  noslip_iterations=str(physics.NOSLIP_ITERATIONS))
    defaults = ET.SubElement(root, 'default')
    default_geom = dict(friction=physics.DEFAULT_FRICTION,
                        solref=physics.CONTACT_SOLREF, solimp=physics.CONTACT_SOLIMP)
    if mjx_compatible:
        # GPU批量仿真不需要机器人自身各连杆之间碰撞。用互补掩码只保留
        # “机器人-地形”接触，可显著减少MJX静态分配的候选接触数量。
        default_geom.update(contype='1', conaffinity='2')
    ET.SubElement(defaults, 'geom', **default_geom)
    ET.SubElement(defaults, 'joint', damping=str(physics.DEFAULT_JOINT_DAMPING),
                  armature=str(physics.DEFAULT_JOINT_ARMATURE))
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth='1400', offheight='1000')
    ET.SubElement(visual, 'headlight', ambient='0.5 0.5 0.5')
    world = ET.SubElement(root, 'worldbody')
    ET.SubElement(world, 'light', pos='0 -1 2')
    add_terrain(root, world, terrain, slope_deg=slope_deg, obstacle_mm=obstacle_mm)
    if mjx_compatible:
        for geom in world.findall('./geom'):
            if geom.get('contype') != '0':
                geom.set('contype', '2')
                geom.set('conaffinity', '1')
    if ground_quat is not None:
        if terrain != 'flat':
            raise ValueError('站立平台不能与路线地形组合')
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, np.asarray(ground_quat, dtype=float))
        normal = rotation.reshape(3, 3)[:, 2]
        # 平台实体厚40 mm，覆盖1.6×1.6 m。
        # 仅平台倾斜，原floor保持世界水平，平台最低角也高于地面。
        platform_half_height = physics.PLATFORM_HALF_SIZE[2]
        center = np.array([0., 0., physics.PLATFORM_TOP_Z]) - normal * platform_half_height
        platform_attrs = {}
        if mjx_compatible:
            platform_attrs.update(contype='2', conaffinity='1')
        ET.SubElement(world, 'geom', name='stand_platform', type='box',
                      size=' '.join(map(str, physics.PLATFORM_HALF_SIZE)),
                      pos=' '.join(map(str, center)), quat=' '.join(map(str, ground_quat)),
                      rgba='.35 .55 .8 1', group='2', friction=physics.PLATFORM_FRICTION,
                      **platform_attrs)
    # Place the lowest capsule surface on the ground in HOME.
    height = physics.BODY_HOME_HEIGHT + (
        physics.FOOT_BALL_RADIUS + physics.INITIAL_GROUND_CLEARANCE if dynamic else 0.006)
    torso = ET.SubElement(world, 'body', name='body', pos=f'0 0 {height}')
    if dynamic:
        ET.SubElement(torso, 'freejoint', name='floating_body')
    if imu:
        # 暂设IMU位于机身原点、轴向与机身一致；不增加质量或碰撞。
        ET.SubElement(torso, 'site', name='imu_site', pos='0 0 0', size='.003')
        sensors = ET.SubElement(root, 'sensor')
        ET.SubElement(sensors, 'accelerometer', name='imu_accel', site='imu_site')
        ET.SubElement(sensors, 'gyro', name='imu_gyro', site='imu_site')
    ET.SubElement(torso, 'geom', name='body_collision', type='cylinder',
                  size=f'{physics.BODY_RADIUS} {physics.BODY_HALF_HEIGHT}',
                  mass=str(physics.BODY_MASS), rgba='0.65 0.69 0.75 1')
    actuators = ET.SubElement(root, 'actuator') if dynamic else None
    home = []
    for i, leg in enumerate(LEGS):
        coxa, femur, tibia = np.array([COXA, FEMUR, TIBIA]) + errors.length_mm[i] / 1000
        parent = ET.SubElement(torso, 'body', name=f'leg{i}_mount',
                               pos=' '.join(map(str, np.asarray(leg.mount) + errors.mount_mm[i] / 1000)),
                               euler=f'0 0 {np.deg2rad(leg.yaw_deg)}')
        for j, (part, length, radius, axis, color, offset) in enumerate((
            ('coxa', coxa, physics.LINK_RADII[0], '0 0 1', '0.2 0.65 0.9 1', 0),
            ('femur', femur, physics.LINK_RADII[1], '0 -1 0', '0.95 0.65 0.2 1', coxa),
            ('tibia', tibia, physics.LINK_RADII[2], '0 -1 0', '0.35 0.8 0.45 1', femur),
        )):
            parent = ET.SubElement(parent, 'body', name=f'leg{i}_{part}', pos=f'{offset} 0 0')
            joint_name = f'leg{i}_{part}_joint'
            limits = geometric_limits(i)[j]
            ET.SubElement(parent, 'joint', name=joint_name, axis=axis,
                          limited='true', range=' '.join(map(str, limits)))
            ET.SubElement(parent, 'geom', type='capsule', fromto=f'0 0 0 {length} 0 0',
                          size=str(radius), mass=str(physics.LINK_MASSES[j]), rgba=color)
            if dynamic:
                ET.SubElement(actuators, 'position', name=f'leg{i}_{part}_servo', joint=joint_name,
                              kp=str(physics.POSITION_KP), kv=str(physics.POSITION_KV),
                              forcelimited='true', forcerange=f'-{physics.MAX_TORQUE} {physics.MAX_TORQUE}',
                              ctrllimited='true', ctrlrange=' '.join(map(str, limits)))
        # 球头仅作连接外观；物理等效球窝由无驱动ball joint实现，避免凹球壳碰撞不稳定。
        ET.SubElement(parent, 'geom', name=f'foot{i}_ball_visual', type='sphere',
                      pos=f'{tibia} 0 0', size=str(physics.FOOT_BALL_RADIUS), mass='0',
                      contype='0', conaffinity='0', rgba='1 0.3 0.15 1')
        if dynamic:
            nominal_q = inverse(i, np.asarray(leg.mount) + leg.home)
            parent_rotation = frames(i, nominal_q)['shin'][:3, :3]
            relative_rotation = parent_rotation.T  # 零球关节时圆盘法向在HOME中朝世界+Z。
            relative_quat = np.empty(4)
            mujoco.mju_mat2Quat(relative_quat, relative_rotation.reshape(-1))
            pad = ET.SubElement(parent, 'body', name=f'foot{i}_pad', pos=f'{tibia} 0 0',
                                quat=' '.join(map(str, relative_quat)))
            ET.SubElement(pad, 'joint', name=f'foot{i}_ball_joint', type='ball', limited='true',
                          range=f'0 {np.deg2rad(physics.FOOT_PAD_MAX_TILT_DEG)}',
                          damping=str(physics.FOOT_PAD_DAMPING))
            if mjx_compatible:
                # MJX解析碰撞版本：保留20 mm支撑半径和球心下方8 mm的底面。
                # 厚度取原拱顶高度，圆柱中心相对球头向下平移半个厚度。
                half_height = physics.FOOT_PAD_DOME_HEIGHT / 2
                center_z = -physics.FOOT_BALL_RADIUS + half_height
                ET.SubElement(pad, 'geom', name=f'foot{i}_collision', type='cylinder',
                              pos=f'0 0 {center_z}',
                              size=f'{physics.FOOT_PAD_RADIUS} {half_height}',
                              mass=str(physics.FOOT_PAD_MASS), rgba='0.85 0.25 0.12 1',
                              friction=physics.FOOT_PAD_FRICTION)
            else:
                # CPU MuJoCo版本保留封闭球面拱顶网格。
                ET.SubElement(pad, 'geom', name=f'foot{i}_collision', type='mesh',
                              mesh='foot_pad_dome',
                              mass=str(physics.FOOT_PAD_MASS), rgba='0.85 0.25 0.12 1',
                              friction=physics.FOOT_PAD_FRICTION)
            ET.SubElement(pad, 'site', name=f'foot{i}_contact', pos=f'0 0 {-physics.FOOT_BALL_RADIUS}',
                          size='.002', rgba='1 0.8 0.1 1')
        ET.SubElement(parent, 'site', name=f'foot{i}', pos=f'{tibia} 0 0',
                      size='0.007', rgba='1 0.3 0.15 1')
        home.extend(inverse(i, np.asarray(leg.mount) + leg.home))
    if not mjx_compatible:
        _add_foot_pad_mesh(root)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    data = mujoco.MjData(model)
    home = errors.joint_targets(home)
    limits = controlled_joint_ranges(model)
    home = np.clip(home, limits[:, 0], limits[:, 1])
    set_controlled_qpos(model, data, np.asarray(home))
    if dynamic:
        data.ctrl[:] = home
    mujoco.mj_forward(model, data)
    if not any(np.any(a) for a in (errors.servo_deg, errors.mount_mm, errors.length_mm)):
        for i, leg in enumerate(LEGS):
            np.testing.assert_allclose(data.site(f'foot{i}').xpos,
                                       np.asarray(leg.mount) + leg.home + [0, 0, height], atol=1e-12)
    if dynamic:
        # 仅抬高初始整体位置防止足球穿地，不修正各腿误差；之后自由落地。
        lowest = min(data.site(f'foot{i}').xpos[2] - physics.FOOT_BALL_RADIUS for i in range(6))
        data.qpos[2] += max(0., physics.INITIAL_GROUND_CLEARANCE - lowest)
        mujoco.mj_forward(model, data)
    return model, data
