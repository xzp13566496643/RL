"""地形构建：生成带碰撞的平地、斜坡、障碍物和组合路线，尺寸单位为米。"""
import math
import xml.etree.ElementTree as ET

import numpy as np

from . import physics

TERRAINS = ('flat', 'ramp', 'obstacles', 'course', 'steep', 'stairs', 'atlas')
STEEP_SLOPES = (10, 15, 20)
STAIR_HEIGHT = 0.050  # 每级50mm，约为HOME机身原点离地高度的一半。
STAIR_TREAD = 0.25

# 站立/训练地形图集：4列类型×4行难度，所有并行环境可共享同一MjModel。
ATLAS_TYPES = ('flat', 'pyramid', 'rough', 'obstacles')
ATLAS_LEVELS = 4
ATLAS_PATCH_SIZE = 2.0
ATLAS_PATCH_SPACING = 2.4
# 图集基准面与世界地面齐平，离开地形块后不会撞到0.7 m高的盒子外壁。
ATLAS_TOP_Z = 0.0
ATLAS_SLOPE_DEG = (3., 7., 11., 15.)
ATLAS_ROUGH_M = (.003, .007, .011, .015)
ATLAS_OBSTACLE_M = (.005, .010, .015, .020)


def atlas_patch_center(kind, level):
    """返回指定地形块中心XY；布局关于世界原点对称。"""
    col = ATLAS_TYPES.index(kind)
    if not 0 <= level < ATLAS_LEVELS:
        raise ValueError('地形难度超出图集范围')
    return np.array(((col - (len(ATLAS_TYPES)-1)/2) * ATLAS_PATCH_SPACING,
                     (level - (ATLAS_LEVELS-1)/2) * ATLAS_PATCH_SPACING))


def add_training_atlas(root, world):
    """加入固定训练图集；返回值由reset按块索引选择出生点。"""
    half, base_depth = ATLAS_PATCH_SIZE/2, .06

    def box(name, center, half_size, top, rgba):
        hx, hy = half_size
        height = top - (ATLAS_TOP_Z - base_depth)
        ET.SubElement(world, 'geom', name=name, type='box',
                      pos=f'{center[0]} {center[1]} {ATLAS_TOP_Z-base_depth+height/2}',
                      size=f'{hx} {hy} {height/2}', rgba=rgba, group='2',
                      friction=physics.PLATFORM_FRICTION)

    for level in range(ATLAS_LEVELS):
        # 平地。
        center = atlas_patch_center('flat', level)
        box(f'atlas_flat_{level}', center, (half, half), ATLAS_TOP_Z, '.32 .48 .62 1')

        # 四个坡面围绕中央平台；局部坡面平整，整体是Eigenbot式金字塔坡地。
        center = atlas_patch_center('pyramid', level)
        angle = math.radians(ATLAS_SLOPE_DEG[level])
        run, plateau, width, thick = .65, .35, .82, .02
        rise = run * math.tan(angle)
        for suffix, dx, dy, euler in (
            ('west', -(plateau + run/2), 0., (0., -angle, 0.)),
            ('east', +(plateau + run/2), 0., (0., +angle, 0.)),
            ('south', 0., -(plateau + run/2), (+angle, 0., 0.)),
            ('north', 0., +(plateau + run/2), (-angle, 0., 0.)),
        ):
            sx, sy = ((run/2, width) if dx else (width, run/2))
            ET.SubElement(world, 'geom', name=f'atlas_pyramid_{level}_{suffix}', type='box',
                          pos=f'{center[0]+dx} {center[1]+dy} {ATLAS_TOP_Z+rise/2-thick}',
                          size=f'{sx} {sy} {thick}', euler=' '.join(map(str, euler)),
                          rgba='.34 .56 .74 1', group='2', friction=physics.PLATFORM_FRICTION)
        box(f'atlas_pyramid_{level}_center', center, (plateau, plateau),
            ATLAS_TOP_Z + rise, '.38 .62 .78 1')

        # 固定随机高度砖阵列，模拟轻微高度场；不同level使用不同幅度。
        center = atlas_patch_center('rough', level)
        rng = np.random.default_rng(1000 + level)
        cells, cell = 8, ATLAS_PATCH_SIZE/8
        # 世界平面位于Z=0；使用非负起伏，避免负高度砖被平面覆盖。
        heights = rng.uniform(0., 2*ATLAS_ROUGH_M[level], (cells, cells))
        for iy in range(cells):
            for ix in range(cells):
                xy = center + [(ix+.5)*cell-half, (iy+.5)*cell-half]
                box(f'atlas_rough_{level}_{iy}_{ix}', xy, (cell/2, cell/2),
                    ATLAS_TOP_Z + heights[iy, ix], '.42 .50 .38 1')

        # 平底加离散矮障碍；尺寸按当前约100mm机身高度缩放。
        center = atlas_patch_center('obstacles', level)
        box(f'atlas_obstacles_{level}_base', center, (half, half), ATLAS_TOP_Z, '.48 .43 .34 1')
        rng = np.random.default_rng(2000 + level)
        for i in range(12):
            xy = center + rng.uniform(-.7, .7, 2)
            sx, sy = rng.uniform(.06, .16, 2)
            box(f'atlas_obstacles_{level}_{i}', xy, (sx, sy),
                ATLAS_TOP_Z + rng.uniform(.5, 1.) * ATLAS_OBSTACLE_M[level], '.70 .48 .22 1')


def add_terrain(root, world, kind='flat', slope_deg=5.0, obstacle_mm=8.0):
    if kind not in TERRAINS:
        raise ValueError(f'Unknown terrain: {kind}')
    if not 0 < slope_deg <= 20 or not 0 < obstacle_mm <= 100:
        raise ValueError('slope must be in (0,20] deg, obstacle height in (0,100] mm')
    asset = ET.SubElement(root, 'asset')
    ET.SubElement(asset, 'texture', name='grid', type='2d', builtin='checker',
                  width='256', height='256', rgb1='0.19 0.23 0.27', rgb2='0.25 0.29 0.33')
    ET.SubElement(asset, 'material', name='ground', texture='grid', texrepeat='10 10', texuniform='true')
    ET.SubElement(world, 'geom', name='floor', type='plane', size='8 8 0.1', material='ground', group='2')
    for name, pos, size, color in (
        ('world_x', '0.30 -0.42 0.001', '0.15 0.004 0.001', '0.9 0.2 0.15 1'),
        ('world_y', '-0.42 0.30 0.001', '0.004 0.15 0.001', '0.2 0.9 0.2 1'),
    ):
        ET.SubElement(world, 'geom', name=name, type='box', pos=pos, size=size,
                      rgba=color, contype='0', conaffinity='0')

    if kind == 'atlas':
        add_training_atlas(root, world)
        return

    def box(name, x, length, height, color):
        ET.SubElement(world, 'geom', name=name, type='box', pos=f'{x} 0 {height/2}',
                      size=f'{length/2} 0.35 {height/2}', rgba=color, group='2')

    if kind in ('obstacles', 'course'):
        h = obstacle_mm / 1000
        box('obstacle_1', 0.50, 0.06, h, '0.95 0.6 0.15 1')
        box('obstacle_2', 0.80, 0.08, h * 1.5, '0.9 0.4 0.12 1')
    if kind == 'stairs':
        # Three 50mm rises, a 500mm landing, then three equal drops to the floor.
        start = 0.50
        for i, (level, length) in enumerate(((1, STAIR_TREAD), (2, STAIR_TREAD),
                                            (3, 2*STAIR_TREAD), (2, STAIR_TREAD),
                                            (1, STAIR_TREAD))):
            color = '0.85 0.55 0.2 1' if i % 2 == 0 else '0.7 0.4 0.15 1'
            box(f'stair_{i+1}', start + length/2, length, level*STAIR_HEIGHT, color)
            start += length
    def ramp(prefix, x0, length, platform, angle, color):
        h = length * math.tan(math.radians(angle))
        # Convex wedges with flush entrances, a platform, then a descending wedge.
        for name, start, z0, z1 in (
            (prefix + '_up', x0, 0, h), (prefix + '_down', x0 + length + platform, h, 0),
        ):
            vertices = [(x, y, z) for x, top in ((0, z0), (length, z1))
                        for y in (-0.35, 0.35) for z in (-0.015, top)]
            ET.SubElement(asset, 'mesh', name=name + '_mesh',
                          vertex=' '.join(str(v) for point in vertices for v in point))
            ET.SubElement(world, 'geom', name=name, type='mesh', mesh=name + '_mesh',
                          pos=f'{start} 0 0', rgba=color, group='2')
        box(prefix + '_platform', x0 + length + platform/2, platform, h, color)

    if kind in ('ramp', 'course'):
        ramp('ramp', 0.50 if kind == 'ramp' else 1.15, 0.60, 0.35,
             slope_deg, '0.3 0.5 0.75 1')
    if kind == 'steep':
        for i, (angle, color) in enumerate(zip(STEEP_SLOPES, (
            '0.95 0.7 0.15 1', '0.95 0.4 0.12 1', '0.8 0.15 0.15 1',
        ))):
            # 0.35m ascent + 0.25m platform + 0.35m descent, then 0.30m flat.
            ramp(f'steep_{angle}', 0.50 + i*1.25, 0.35, 0.25, angle, color)
