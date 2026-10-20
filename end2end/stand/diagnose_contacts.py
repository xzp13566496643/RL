"""碰撞诊断：复现固定初始动作，逐物理步统计地形接触、穿透和警告，不改参数。"""
import argparse
import json

import mujoco
import numpy as np

from hexapod_sim.perturbations import add_arguments, from_arguments
from . import scene
from .env import StandEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=5.)
    add_arguments(parser)
    parser.set_defaults(joint_bias_deg=3.)
    scene.add_arguments(parser)
    args = parser.parse_args()
    from_arguments(args, parser)
    if not np.isfinite(args.seconds) or args.seconds <= 0:
        parser.error('seconds必须为有限正数')
    try:
        env = StandEnv(joint_bias_deg=args.joint_bias_deg, mount_bias_mm=args.mount_bias_mm,
                       length_bias_mm=args.length_bias_mm, scene_config=scene.from_arguments(args))
        env.reset(args.seed)
    except ValueError as exc:
        parser.error(str(exc))
    model, data = env.model, env.data
    terrain = {i for i in range(model.ngeom)
               if model.geom_bodyid[i] == 0 and model.geom_group[i] == 2}
    disabled = int(model.opt.disableflags)
    print(json.dumps(dict(seed=args.seed, slope_deg=env.slope_deg, timestep=model.opt.timestep,
                          contact_disabled=bool(disabled & int(mujoco.mjtDisableBit.mjDSBL_CONTACT)),
                          constraints_disabled=bool(disabled & int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)),
                          terrain_type=env.patch['kind'], terrain_level=env.patch['level'],
                          terrain_friction=sorted({tuple(model.geom_friction[i]) for i in terrain}),
                          geoms=[dict(id=i, name=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i),
                                      contype=int(model.geom_contype[i]), conaffinity=int(model.geom_conaffinity[i]),
                                      body=int(model.geom_bodyid[i])) for i in range(model.ngeom)]), indent=2), flush=True)
    seen = set()
    worst = None
    for step in range(int(np.ceil(args.seconds / model.opt.timestep))):
        # 与demo默认相同：保持reset设置的初始ctrl，不写qpos，不修改接触参数。
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            print('NONFINITE STATE', flush=True)
            break
        contact_details = []
        for c in data.contact:
            if c.geom1 not in terrain and c.geom2 not in terrain:
                continue
            other = c.geom2 if c.geom1 in terrain else c.geom1
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other) or f'geom#{other}'
            seen.add(name)
            contact_details.append(dict(other=name, distance_m=float(c.dist), active=bool(c.efc_address >= 0)))
            if worst is None or c.dist < worst['distance_m']:
                worst = dict(time_s=float(data.time), other=name, distance_m=float(c.dist))
        if step % max(1, round(.25 / model.opt.timestep)) == 0:
            print(json.dumps(dict(time_s=float(data.time), clearance_m=env.state()['clearance_m'],
                                  tilt_deg=env.state()['tilt_deg'], terrain_contacts=contact_details)), flush=True)
    print(json.dumps(dict(terrain_contact_geoms=sorted(seen), worst_contact=worst,
                          warnings=[dict(index=i, count=int(w.number)) for i, w in enumerate(data.warning) if w.number]),
                     indent=2), flush=True)


if __name__ == '__main__':
    main()
