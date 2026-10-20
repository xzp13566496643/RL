"""ARC-RL Tick on mjbatch: bounded PPO smoke training and a native viewer.

The original MJCF is loaded unchanged. The small 67-D environment and PPO loop
are a local smoke-test adapter, not ARC-RL's original SAC training experiment.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import mujoco
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'vendor/mjbatch/examples'))
from mjbatch import Batch
from window import Window

MODEL = ROOT / 'vendor/ARC_RL/src/models/tick_v1.xml'
OUT = ROOT / 'artifacts/tick-mjbatch'
FEET = ('l1_foot', 'r1_foot', 'l2_foot', 'r2_foot', 'l3_foot', 'r3_foot')
PHASE = np.array([np.pi, 0, 0, np.pi, np.pi, 0])
SUBSTEPS, HORIZON, OBS_DIM, ACT_DIM = 25, 32, 67, 18


class TickBatch:
    def __init__(self, n, seed=0):
        self.model = mujoco.MjModel.from_xml_path(str(MODEL))
        assert (self.model.nq, self.model.nv, self.model.nu) == (25, 24, 18)
        self.batch = Batch(self.model, n, num_threads=min(8, os.cpu_count() or 1), forward=True)
        self.n, self.rng = n, np.random.default_rng(seed)
        self.q, self.v, self.u = (self.batch.bind(f) for f in ('qpos', 'qvel', 'ctrl'))
        self.positions = self.batch.bind('xpos')
        self.feet = [self.model.body(f).id for f in FEET]
        self.warning = self.batch.bind('warning')
        self.dt = self.model.opt.timestep * SUBSTEPS
        self.age = np.zeros(n, dtype=np.int64)
        self.prev = np.zeros((n, ACT_DIM), np.float32)
        self.initial = self.model.numeric_data[:self.model.nq].copy()
        self.initial[7:] = np.deg2rad(self.initial[7:])
        self.reset(np.arange(n))

    def reset(self, ids):
        ids = np.asarray(ids, dtype=np.int64)
        self.batch.reset(ids)
        self.q[ids] = self.initial
        self.q[ids, 7:] += self.rng.uniform(-0.03, 0.03, (len(ids), ACT_DIM))
        self.q[ids, 2] += self.rng.uniform(0, 0.005, len(ids))
        self.v[ids], self.u[ids], self.prev[ids], self.age[ids] = 0, 0, 0, 0
        self.batch.forward(ids)

    def obs(self):
        angle = 2 * np.pi * 1.5 * self.age * self.dt
        return np.concatenate((self.q[:, 2:], np.clip(self.v, -20, 20) / 5,
            self.prev, np.sin(angle)[:, None], np.cos(angle)[:, None]), axis=1).astype(np.float32)

    def step(self, action):
        action = np.clip(action, -1, 1).astype(np.float32)
        before = self.q[:, 0].copy()
        self.u[:] = action
        self.batch.step(nstep=SUBSTEPS)
        self.age += 1
        velocity = (self.q[:, 0] - before) / self.dt
        angle = 2 * np.pi * 1.5 * self.age * self.dt
        stance = (angle[:, None] + PHASE) % (2 * np.pi) < 0.6 * 2 * np.pi
        # Same height heuristic as upstream TickEnv; not force-based contact sensing.
        actual = self.positions[:, self.feet, 2] < 0.06
        errors = np.sum(stance != actual, axis=1)
        healthy = (self.q[:, 2] >= 0.08) & (self.q[:, 2] <= 0.35)
        finite = np.isfinite(self.q).all(1) & np.isfinite(self.v).all(1)
        reward = (1.5 * np.maximum(0, 1 - np.abs(velocity - 0.33) / 0.33)
            + healthy + 1.5 * (1 - errors / 6) - 0.1 * errors
            - 0.05 * np.sum(action**2, 1)
            - 0.02 * np.sum((action - self.prev)**2, 1)
            - 0.05 * np.sum(self.v[:, 3:5]**2, 1)
            - 0.02 * self.v[:, 2]**2 - 0.001 * np.sum(self.q[:, 7:]**2, 1))
        self.prev[:] = action
        terminated = ~healthy | ~finite
        done = terminated | (self.age >= 200)
        assert finite.all(), 'Non-finite physics state'
        return reward.astype(np.float32), done, terminated


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        def mlp(out):
            return nn.Sequential(nn.Linear(OBS_DIM, 64), nn.Tanh(),
                nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, out))
        self.actor, self.critic = mlp(ACT_DIM), mlp(1)
        self.log_std = nn.Parameter(torch.full((ACT_DIM,), -1.8))
        nn.init.normal_(self.actor[-1].weight, std=0.01)
        nn.init.zeros_(self.actor[-1].bias)

    def forward(self, obs):
        return Normal(self.actor(obs), self.log_std.clamp(-3, 0).exp()), self.critic(obs).squeeze(-1)

    @torch.no_grad()
    def act(self, env):
        return self(torch.from_numpy(env.obs()))[0].mean.numpy()


def collect(env, net):
    buf = {k: [] for k in ('obs', 'act', 'logp', 'value', 'reward', 'done', 'terminal', 'next_value')}
    episodes = 0
    with torch.no_grad():
        for _ in range(HORIZON):
            obs = torch.from_numpy(env.obs())
            dist, value = net(obs)
            action = dist.sample()
            reward, done, terminal = env.step(action.numpy())
            next_value = net(torch.from_numpy(env.obs()))[1]
            values = (obs, action, dist.log_prob(action).sum(-1), value,
                torch.from_numpy(reward), torch.from_numpy(done), torch.from_numpy(terminal), next_value)
            for key, val in zip(buf, values):
                buf[key].append(val.clone())
            ids = np.flatnonzero(done)
            episodes += len(ids)
            if len(ids):
                env.reset(ids)
    b = {k: torch.stack(v) for k, v in buf.items()}
    advantage, carry = torch.zeros_like(b['reward']), torch.zeros(env.n)
    for t in reversed(range(HORIZON)):
        delta = b['reward'][t] + 0.99 * (~b['terminal'][t]) * b['next_value'][t] - b['value'][t]
        carry = delta + 0.99 * 0.95 * (~b['done'][t]) * carry
        advantage[t] = carry
    b['return'] = advantage + b['value']
    b['advantage'] = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
    return {k: v.flatten(0, 1) for k, v in b.items()}, episodes


def update(net, opt, b):
    losses = []
    for _ in range(3):
        for ids in torch.randperm(len(b['obs'])).split(256):
            dist, value = net(b['obs'][ids])
            ratio = (dist.log_prob(b['act'][ids]).sum(-1) - b['logp'][ids]).exp()
            adv = b['advantage'][ids]
            loss = (-torch.minimum(ratio * adv, ratio.clamp(0.8, 1.2) * adv).mean()
                + 0.5 * (value - b['return'][ids]).square().mean() - 0.001 * dist.entropy().sum(-1).mean())
            assert torch.isfinite(loss), 'Non-finite PPO loss'
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1)
            opt.step()
            losses.append(loss.item())
    return float(np.mean(losses))


def verify():
    env = TickBatch(4, seed=123)
    data = mujoco.MjData(env.model)
    data.qpos[:], data.qvel[:] = env.q[0], env.v[0]
    mujoco.mj_forward(env.model, data)
    rng = np.random.default_rng(22)
    max_error = 0.0
    for _ in range(40):
        controls = rng.uniform(-0.1, 0.1, (4, ACT_DIM))
        env.u[:] = controls
        data.ctrl[:] = controls[0]
        env.batch.step(nstep=SUBSTEPS)
        for _ in range(SUBSTEPS):
            mujoco.mj_step(env.model, data)
        mujoco.mj_forward(env.model, data)
        max_error = max(max_error, float(np.max(np.abs(env.q[0] - data.qpos))))
        np.testing.assert_allclose(env.q[0], data.qpos, atol=1e-10, rtol=1e-10)
    other = env.q[1:].copy()
    env.reset([0])
    np.testing.assert_array_equal(env.q[1:], other)
    assert env.obs().shape == (4, OBS_DIM)
    assert np.isfinite(env.q).all() and not env.warning.any()
    result = {'serial_batch_match': True, 'max_qpos_error': max_error,
        'selective_reset': True, 'finite_states': True, 'warnings': int(env.warning.sum()),
        'nq': env.model.nq, 'nv': env.model.nv, 'actuators': env.model.nu, 'dt': env.dt}
    print('CHECK', json.dumps(result), flush=True)
    return result


class View:
    def __init__(self, model):
        import glfw
        self.glfw = glfw
        self.model, self.data = model, mujoco.MjData(model)
        self.window = Window(model, self.data, title='Tick | mjbatch | short PPO test')
        self.window.camera.azimuth, self.window.camera.elevation = 135, -27

    def show(self, env, title, detail, grid=False, capture=None):
        if not self.window.open():
            return False
        self.window.camera.distance = 5.2 if grid else 1.65
        self.window.camera.lookat[:] = [0, 0, 0.12] if grid else [*env.q[0, :2], 0.12]
        self.data.qpos[:] = env.q[0]
        self.data.qvel[:] = env.v[0]
        self.data.time = env.age[0] * env.dt
        mujoco.mj_forward(self.model, self.data)
        self.window.draw()
        scene = self.window.scene
        if grid:
            # Display offsets only: the independent batch worlds remain unchanged.
            scene.ngeom = 0
            mujoco.mjv_addGeoms(self.model, self.data, self.window.option,
                self.window.perturb, mujoco.mjtCatBit.mjCAT_STATIC, scene)
            for i in range(min(16, env.n)):
                self.data.qpos[:], self.data.qvel[:] = env.q[i], env.v[i]
                self.data.qpos[:2] += [(i % 4 - 1.5) * 1.1, (i // 4 - 1.5) * 1.1]
                mujoco.mj_forward(self.model, self.data)
                mujoco.mjv_addGeoms(self.model, self.data, self.window.option,
                    self.window.perturb, mujoco.mjtCatBit.mjCAT_DYNAMIC, scene)
        viewport = mujoco.MjrRect(0, 0, *self.glfw.get_framebuffer_size(self.window.window))
        mujoco.mjr_render(viewport, scene, self.window.context)
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
            viewport, title + '\n' + detail,
            'ARC-RL Tick | 6 legs / 18 motors\nMuJoCo 3.14.0 | CPU mjbatch', con=self.window.context)
        if capture:
            from PIL import Image
            pixels = np.empty((viewport.height, viewport.width, 3), np.uint8)
            mujoco.mjr_readPixels(pixels, None, viewport, self.window.context)
            Image.fromarray(np.flipud(pixels)).save(OUT / (capture + '.png'))
            print('CAPTURE', capture, flush=True)
        self.glfw.swap_buffers(self.window.window)
        self.glfw.poll_events()
        return True

    def close(self):
        self.window.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--num-envs', type=int, default=64)
    p.add_argument('--updates', type=int, default=30)
    p.add_argument('--seconds', type=float, default=8)
    p.add_argument('--headless', action='store_true')
    p.add_argument('--play', action='store_true', help='Load the locally saved short-test policy')
    p.add_argument('--check', action='store_true', help='Only compare mjbatch against serial MuJoCo')
    args = p.parse_args()
    if args.num_envs < 1 or args.updates < 0 or args.seconds < 0:
        p.error('num-envs must be positive; updates and seconds must be nonnegative')
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(0)
    checks = verify()
    if args.check:
        (OUT / 'checks.json').write_text(json.dumps(checks, indent=2))
        return
    env, net = TickBatch(args.num_envs), Policy()
    checkpoint = OUT / 'tick-short-ppo.pt'
    if args.play:
        net.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    view = None
    records, training = [], time.perf_counter()
    before_weights = torch.cat([x.detach().flatten() for x in net.parameters()]).clone()
    try:
        if not args.headless:
            view = View(env.model)
            view.show(env, 'MODEL LOADED', 'Original MJCF | no model edits', capture='01-model-loaded')
        if not args.play:
            opt = torch.optim.Adam(net.parameters(), lr=3e-4)
            for it in range(args.updates):
                b, episodes = collect(env, net)
                loss = update(net, opt, b)
                record = {'update': it + 1, 'transitions': (it + 1) * HORIZON * env.n,
                    'mean_reward': float(b['reward'].mean()), 'loss': loss, 'episodes_ended': episodes}
                records.append(record)
                print('PPO', json.dumps(record), flush=True)
                if view:
                    tag = ('02-training-start' if it == 0 else '03-training-middle' if it + 1 == max(1, args.updates // 2)
                        else '04-training-complete' if it + 1 == args.updates else None)
                    view.show(env, f'SHORT PPO | update {it+1}/{args.updates}',
                        f'{env.n} parallel worlds | 16 shown | reward {record["mean_reward"]:.3f}', grid=True, capture=tag)
            torch.save(net.state_dict(), checkpoint)
        training_seconds = time.perf_counter() - training
        changed = float(torch.linalg.vector_norm(torch.cat([x.detach().flatten() for x in net.parameters()]) - before_weights))
        if records:
            assert changed > 0 and all(np.isfinite(r['loss']) for r in records)
        training_warnings = int(env.warning.sum())
        env.reset(np.arange(env.n))
        preview = []
        if view:
            deadline, frame = time.perf_counter() + args.seconds, 0
            while time.perf_counter() < deadline and view.window.open():
                reward, done, _ = env.step(net.act(env))
                preview.append(env.q[0].copy())
                tag = '05-policy-preview' if frame == 30 else None
                view.show(env, 'SHORT-TRAINED POLICY PREVIEW',
                    f'Frame {frame} | reward {reward[0]:.3f} | exploratory policy, not converged', capture=tag)
                ids = np.flatnonzero(done)
                if len(ids):
                    env.reset(ids)
                frame += 1
                view.window.pace(env.dt)
        result = {'checks': checks, 'num_envs': env.n, 'threads': env.batch.num_threads,
            'updates': len(records), 'transitions': len(records) * HORIZON * env.n,
            'physics_steps': len(records) * HORIZON * env.n * SUBSTEPS,
            'training_seconds': training_seconds, 'policy_parameter_change_l2': changed,
            'training_warning_count': training_warnings,
            'model_sha256': hashlib.sha256(MODEL.read_bytes()).hexdigest(),
            'training_history': records, 'converged_walking_claim': False}
        if preview:
            np.savez_compressed(OUT / 'preview-trajectory.npz', qpos=np.asarray(preview), dt=env.dt)
        (OUT / ('play-result.json' if args.play else 'result.json')).write_text(json.dumps(result, indent=2))
        print('COMPLETE', json.dumps({k: v for k, v in result.items() if k != 'training_history'}), flush=True)
    finally:
        if view:
            view.close()
        print('CLOSED: Tick viewer and GLFW resources released', flush=True)
    result['viewer_closed'] = True
    (OUT / ('play-result.json' if args.play else 'result.json')).write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
