"""从训练保存的latest-policy.npz导出浏览器演示所需的Actor权重。"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def export_actor(source, destination):
    source = Path(source)
    destination = Path(destination)
    with np.load(source) as values:
        arrays = {
            'format_version': np.array(1, dtype=np.int64),
            'iteration': np.asarray(values['iteration']),
            'total_steps': np.asarray(values['total_steps']),
        }
        for index in range(3):
            arrays[f'actor_w{index}'] = np.asarray(values[f'actor_w{index}'])
            arrays[f'actor_b{index}'] = np.asarray(values[f'actor_b{index}'])
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
    return int(arrays['iteration']), int(arrays['total_steps'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    iteration, steps = export_actor(args.source, args.destination)
    print(f'已导出 {args.destination}：第{iteration}轮，{steps}训练步')


if __name__ == '__main__':
    main()
