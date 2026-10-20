"""辅助工具：静态显示六足HOME模型，也可通过 --render 导出预览图片。"""
import argparse
from pathlib import Path
import time

import mujoco
import mujoco.viewer

from ..model import make_model
from ..perturbations import add_arguments, from_arguments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render', type=Path)
    add_arguments(parser)
    args = parser.parse_args()
    errors = from_arguments(args, parser)
    model, data = make_model(perturbations=errors)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [0, 0, 0.07]
    camera.distance, camera.azimuth, camera.elevation = 0.85, 135, -35
    print('18 joints loaded. Static HOME command preview (including configured errors).', flush=True)
    if args.render:
        from PIL import Image
        with mujoco.Renderer(model, height=1000, width=1400) as renderer:
            renderer.update_scene(data, camera=camera)
            Image.fromarray(renderer.render()).save(args.render)
        return
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.lookat[:] = camera.lookat
            viewer.cam.distance = camera.distance
            viewer.cam.azimuth = camera.azimuth
            viewer.cam.elevation = camera.elevation
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        time.sleep(1)


if __name__ == '__main__':
    main()
