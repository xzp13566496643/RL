"""提供Stage 2a实时MuJoCo仿真的浏览器页面。"""
from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import threading
import time

# 无显示器的训练服务器默认使用EGL；需要其他后端时可在启动前覆盖MUJOCO_GL。
os.environ.setdefault('MUJOCO_GL', 'egl')

import mujoco
from PIL import Image

from .simulator import Stage2Simulator


STATIC_DIR = Path(__file__).with_name('static')


class DemoApplication:
    def __init__(self, policy_path, *, render_width=960, render_height=720):
        self.simulator = Stage2Simulator(policy_path)
        self.render_width = int(render_width)
        self.render_height = int(render_height)
        self.lock = threading.RLock()
        self.frame_condition = threading.Condition()
        self.frame = None
        self.frame_number = 0
        self.error = None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name='mujoco-demo', daemon=True)

    def start(self):
        self.thread.start()

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=3.)

    def command(self, name, speed):
        with self.lock:
            self.simulator.set_command(name, speed)
            return self.simulator.status()

    def reset(self):
        with self.lock:
            self.simulator.reset()
            return self.simulator.status()

    def status(self):
        with self.lock:
            result = self.simulator.status()
            result['render_error'] = self.error
            return result

    def wait_for_frame(self, previous, timeout=2.):
        with self.frame_condition:
            self.frame_condition.wait_for(
                lambda: self.frame_number != previous or self.stop_event.is_set(), timeout)
            return self.frame_number, self.frame

    def _run(self):
        renderer = None
        try:
            renderer = mujoco.Renderer(
                self.simulator.model, height=self.render_height, width=self.render_width)
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.distance = .75
            camera.azimuth = 135.
            camera.elevation = -28.
            next_step = time.monotonic()
            next_render = next_step
            while not self.stop_event.is_set():
                now = time.monotonic()
                if now >= next_step:
                    with self.lock:
                        self.simulator.step()
                    next_step += self.simulator.control_dt
                    if now-next_step > .2:
                        next_step = now
                if now >= next_render:
                    with self.lock:
                        camera.lookat[:] = self.simulator.data.qpos[:3]
                        renderer.update_scene(self.simulator.data, camera=camera)
                        pixels = renderer.render()
                    output = io.BytesIO()
                    Image.fromarray(pixels).save(output, format='JPEG', quality=86)
                    with self.frame_condition:
                        self.frame = output.getvalue()
                        self.frame_number += 1
                        self.frame_condition.notify_all()
                    next_render = now + 1./20.
                time.sleep(max(0., min(next_step, next_render)-time.monotonic()))
        except Exception as exc:  # 将渲染/物理错误展示到网页状态栏。
            self.error = f'{type(exc).__name__}: {exc}'
            with self.frame_condition:
                self.frame_condition.notify_all()
        finally:
            if renderer is not None:
                renderer.close()


class DemoHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, handler, application):
        super().__init__(address, handler)
        self.application = application


class Handler(BaseHTTPRequestHandler):
    server_version = 'HexapodDemo/1.0'

    @property
    def app(self):
        return self.server.application

    def log_message(self, fmt, *args):
        print(f'{self.address_string()} - {fmt % args}', flush=True)

    def _send(self, content, content_type, status=HTTPStatus.OK):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(content)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(content)

    def _json(self, value, status=HTTPStatus.OK):
        self._send(json.dumps(value, ensure_ascii=False).encode(),
                   'application/json; charset=utf-8', status)

    def do_GET(self):
        if self.path in ('/', '/index.html'):
            self._send((STATIC_DIR/'index.html').read_bytes(), 'text/html; charset=utf-8')
        elif self.path == '/api/status':
            self._json(self.app.status())
        elif self.path == '/stream':
            self._stream()
        else:
            self._json({'error': 'not found'}, HTTPStatus.NOT_FOUND)

    def do_POST(self):
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if length > 4096:
                raise ValueError('请求过大')
            body = json.loads(self.rfile.read(length) or b'{}')
            if self.path == '/api/command':
                result = self.app.command(str(body.get('command')),
                                          float(body.get('speed', .15)))
            elif self.path == '/api/reset':
                result = self.app.reset()
            else:
                self._json({'error': 'not found'}, HTTPStatus.NOT_FOUND)
                return
            self._json(result)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._json({'error': str(exc)}, HTTPStatus.BAD_REQUEST)

    def _stream(self):
        self.send_response(HTTPStatus.OK)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        number = -1
        try:
            while not self.app.stop_event.is_set():
                number, frame = self.app.wait_for_frame(number)
                if frame is None:
                    continue
                self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\n')
                self.wfile.write(f'Content-Length: {len(frame)}\r\n\r\n'.encode())
                self.wfile.write(frame+b'\r\n')
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    default_policy = (Path(__file__).resolve().parents[2]
                      / 'artifacts/stage2a_actor_10000.npz')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', type=Path, default=default_policy)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    application = DemoApplication(args.policy)
    server = DemoHTTPServer((args.host, args.port), Handler, application)
    application.start()
    print(f'Stage 2a演示：http://{args.host}:{args.port}', flush=True)
    print(f'策略：{args.policy}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('正在关闭演示服务...', flush=True)
    finally:
        server.shutdown()
        server.server_close()
        application.close()


if __name__ == '__main__':
    main()
