"""持续训练入口检查：短训练、指标、checkpoint与恢复。"""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import torch

from .config import DEFAULT_CONFIG
from .train import train


class TrainTests(unittest.TestCase):
    def test_short_training_and_resume(self):
        config = replace(
            DEFAULT_CONFIG, device='cpu', total_iterations=2, log_interval=10,
            checkpoint_interval=1,
            rollout=replace(DEFAULT_CONFIG.rollout, num_steps=8),
            ppo=replace(DEFAULT_CONFIG.ppo, epochs=1, mini_batch_size=4,
                        target_kl=None))
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / 'run'
            train(config, output_dir=run_dir)
            latest = run_dir / 'latest.pt'
            self.assertTrue(latest.exists())
            lines = (run_dir / 'metrics.jsonl').read_text().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[-1])['iteration'], 2)
            checkpoint = torch.load(latest, map_location='cpu', weights_only=False)
            self.assertEqual(checkpoint['iteration'], 2)
            self.assertIn('optimizer', checkpoint)

            resumed = replace(config, total_iterations=3)
            train(resumed, resume=latest, output_dir=run_dir)
            lines = (run_dir / 'metrics.jsonl').read_text().splitlines()
            self.assertEqual(len(lines), 3)
            self.assertEqual(json.loads(lines[-1])['iteration'], 3)
            checkpoint = torch.load(latest, map_location='cpu', weights_only=False)
            self.assertEqual(checkpoint['iteration'], 3)
            self.assertEqual(checkpoint['total_steps'], 24)


if __name__ == '__main__':
    unittest.main()
