"""站立场景：从固定多地形图集中逐局采样地形块、落点、Yaw和关节扰动。"""
from dataclasses import dataclass

import numpy as np

from hexapod_sim.terrain import (ATLAS_LEVELS, ATLAS_PATCH_SIZE, ATLAS_SLOPE_DEG,
                                 atlas_patch_center)


@dataclass(frozen=True)
class SceneConfig:
    flat_probability: float = .3
    max_slope_deg: float = 15.
    initial_joint_deg: float = 20.
    max_terrain_level: int = 3
    settle_seconds: float = 1.
    landing_timeout_seconds: float = 2.
    landing_min_feet: int = 3

    def __post_init__(self):
        if not np.isfinite(self.flat_probability) or not 0 <= self.flat_probability <= 1:
            raise ValueError('平地概率须在0..1')
        for value in (self.max_slope_deg, self.initial_joint_deg, self.settle_seconds,
                      self.landing_timeout_seconds):
            if not np.isfinite(value) or value < 0:
                raise ValueError('场景扰动幅度须为有限非负数')
        if self.max_slope_deg > 15:
            raise ValueError('当前图集金字塔坡度上限15度')
        if not isinstance(self.max_terrain_level, int) or not 0 <= self.max_terrain_level < ATLAS_LEVELS:
            raise ValueError(f'地形难度须为0..{ATLAS_LEVELS-1}整数')
        if (not isinstance(self.landing_min_feet, int)
                or not 1 <= self.landing_min_feet <= 6):
            raise ValueError('落地判定足端数须为1..6整数')


def sample_spawn(config, rng):
    """从固定图集中采样一块地形和安全出生点；相同随机种子完全复现。"""
    if rng.random() < config.flat_probability:
        kind = 'flat'
    else:
        kind = rng.choice(('pyramid', 'rough', 'obstacles'), p=(.375, .375, .25))
    level = int(rng.integers(0, config.max_terrain_level + 1))
    if kind == 'pyramid':
        allowed = [i for i, angle in enumerate(ATLAS_SLOPE_DEG)
                   if i <= config.max_terrain_level and angle <= config.max_slope_deg]
        level = int(rng.choice(allowed)) if allowed else 0
    center = atlas_patch_center(kind, level)
    local = rng.uniform(-.22, .22, 2)
    if kind == 'pyramid':
        side = int(rng.integers(0, 4))
        along = rng.uniform(-.08, .08)
        across = rng.uniform(-.20, .20)
        local = ((-.675 + along, across), (.675 + along, across),
                 (across, -.675 + along), (across, .675 + along))[side]
        local = np.asarray(local)
    return dict(kind=kind, level=level, patch_center=center,
                spawn_xy=center + local, yaw_rad=float(rng.uniform(-np.pi, np.pi)),
                patch_half_extent=ATLAS_PATCH_SIZE/2)


def add_arguments(parser):
    parser.add_argument('--flat', action='store_true', help='使用平地并关闭初始大关节扰动，保留装配误差')
    parser.add_argument('--max-slope-deg', type=float, default=15.)
    parser.add_argument('--flat-probability', type=float, default=.3, help='每局平地概率，其余为坡面/起伏/障碍')
    parser.add_argument('--initial-joint-deg', type=float, default=20.)
    parser.add_argument('--max-terrain-level', type=int, default=3)
    parser.add_argument('--settle-seconds', type=float, default=1.,
                        help='检测到足端支撑后继续保持随机初始目标的时长')
    parser.add_argument('--landing-timeout-seconds', type=float, default=2.)
    parser.add_argument('--landing-min-feet', type=int, default=3)


def from_arguments(args):
    return (SceneConfig(1., 0., 0., 0, args.settle_seconds,
                        args.landing_timeout_seconds, args.landing_min_feet) if args.flat else
            SceneConfig(flat_probability=args.flat_probability, max_slope_deg=args.max_slope_deg,
                        initial_joint_deg=args.initial_joint_deg,
                        max_terrain_level=args.max_terrain_level,
                        settle_seconds=args.settle_seconds,
                        landing_timeout_seconds=args.landing_timeout_seconds,
                        landing_min_feet=args.landing_min_feet))
