"""把训练JSONL指标绘制成无需额外依赖的SVG折线图。"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np


COLORS = ('#2563eb', '#dc2626', '#16a34a', '#9333ea', '#ea580c', '#0891b2')


def read_records(path):
    records = []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _segments(points):
    segment = []
    for point in points:
        if point is None:
            if segment:
                yield segment
                segment = []
        else:
            segment.append(point)
    if segment:
        yield segment


def _panel(records, series, *, x, y, width, height, title):
    values = [float(record[key]) for _, key in series for record in records
              if record.get(key) is not None and np.isfinite(record[key])]
    if not values:
        values = [0., 1.]
    low, high = min(values), max(values)
    padding = (max(abs(low) * .1, .1) if np.isclose(low, high)
               else (high - low) * .08)
    low, high = low - padding, high + padding
    iterations = [record['iteration'] for record in records]
    xmin, xmax = min(iterations), max(iterations)
    if xmin == xmax:
        xmax = xmin + 1
    left, right, top, bottom = x + 65, x + width - 20, y + 38, y + height - 42

    def sx(value):
        return left + (value - xmin) / (xmax - xmin) * (right - left)

    def sy(value):
        return bottom - (value - low) / (high - low) * (bottom - top)

    out = [f'<text x="{x + 12}" y="{y + 22}" class="title">{html.escape(title)}</text>']
    for fraction in (0., .25, .5, .75, 1.):
        py = bottom - fraction * (bottom - top)
        value = low + fraction * (high - low)
        out.append(f'<line x1="{left}" y1="{py:.1f}" x2="{right}" y2="{py:.1f}" class="grid"/>')
        out.append(f'<text x="{left - 7}" y="{py + 4:.1f}" text-anchor="end" class="tick">{value:.3g}</text>')
    out.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" class="axis"/>')
    out.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" class="axis"/>')
    out.append(f'<text x="{left}" y="{bottom + 18}" class="tick">{xmin}</text>')
    out.append(f'<text x="{right}" y="{bottom + 18}" text-anchor="end" class="tick">{max(iterations)}</text>')
    for index, (label, key) in enumerate(series):
        color = COLORS[index % len(COLORS)]
        points = [(sx(record['iteration']), sy(float(record[key])))
                  if record.get(key) is not None and np.isfinite(record[key]) else None
                  for record in records]
        for segment in _segments(points):
            encoded = ' '.join(f'{px:.1f},{py:.1f}' for px, py in segment)
            out.append(f'<polyline points="{encoded}" fill="none" stroke="{color}" stroke-width="2"/>')
        legend_x = left + index * 155
        out.append(f'<line x1="{legend_x}" y1="{top + 3}" x2="{legend_x + 18}" y2="{top + 3}" stroke="{color}" stroke-width="3"/>')
        out.append(f'<text x="{legend_x + 23}" y="{top + 7}" class="legend">{html.escape(label)}</text>')
    return '\n'.join(out)


def write_training_plot(metrics_path, output_path=None, *, command_mode=None,
                        last_records=None):
    metrics_path = Path(metrics_path)
    records = read_records(metrics_path)
    if last_records is not None:
        if last_records <= 0:
            raise ValueError('last_records必须为正整数')
        records = records[-last_records:]
    if not records:
        return None
    output_path = Path(output_path) if output_path else metrics_path.with_name('training_curves.svg')
    if command_mode not in (None, 'stationary', 'moving'):
        raise ValueError('command_mode必须为stationary、moving或None')
    # 每项奖励率独占一个坐标轴，避免不同数量级互相遮挡。
    reward_series = (
        ('Linear Velocity', 'reward_rate_linear_velocity'),
        ('Yaw Velocity', 'reward_rate_yaw_velocity'),
        ('Body Height', 'reward_rate_body_height'),
        ('Body Tilt', 'reward_rate_body_tilt'),
        ('Feet Phase', 'reward_rate_feet_phase'),
        ('Feet Phase No-Lift', 'reward_rate_feet_phase_no_lift'),
        ('Adjacent Coordination', 'reward_rate_adjacent_coordination'),
        ('Feet Air Time', 'reward_rate_feet_air_time'),
        ('Feet Height', 'reward_rate_feet_height'),
        ('HOME L2', 'reward_rate_home_l2'),
        ('HOME L1 Stationary', 'reward_rate_home_l1_stationary'),
        ('Action Rate', 'reward_rate_action_rate'),
        ('Action Magnitude', 'reward_rate_action_magnitude'),
        ('Joint Acceleration', 'reward_rate_joint_acceleration'),
        ('Torque', 'reward_rate_torque'),
        ('Roll / Pitch Rate', 'reward_rate_roll_pitch_rate'),
        ('Vertical Velocity', 'reward_rate_vertical_velocity'),
        ('Feet Slip', 'reward_rate_feet_slip'),
    )
    prefix = f'{command_mode}_' if command_mode else ''
    mode_title = ({'stationary': 'Stationary command',
                   'moving': 'Non-zero velocity command'}.get(command_mode, 'All commands'))
    panels = [(f'{mode_title} / {label} reward rate',
               ((label, f'{prefix}{key}'),)) for label, key in reward_series]
    width, panel_height = 1100, 250
    body = [_panel(records, series, x=0, y=index * panel_height,
                   width=width, height=panel_height, title=title)
            for index, (title, series) in enumerate(panels)]
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{len(panels) * panel_height}" viewBox="0 0 {width} {len(panels) * panel_height}">
<style>.axis{{stroke:#111;stroke-width:1}} .grid{{stroke:#ddd;stroke-width:1}} text{{font-family:sans-serif;fill:#111}} .title{{font-size:16px;font-weight:bold}} .tick{{font-size:11px}} .legend{{font-size:11px}}</style>
<rect width="100%" height="100%" fill="white"/>
{''.join(body)}
</svg>'''
    temporary = output_path.with_suffix(output_path.suffix + '.tmp')
    temporary.write_text(svg, encoding='utf-8')
    temporary.replace(output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path, help='metrics.jsonl文件或训练输出目录')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--mode', choices=('stationary', 'moving'))
    parser.add_argument('--last', type=int, metavar='N',
                        help='只绘制最后N条迭代记录')
    args = parser.parse_args()
    metrics = args.path / 'metrics.jsonl' if args.path.is_dir() else args.path
    output = write_training_plot(metrics, args.output, command_mode=args.mode,
                                 last_records=args.last)
    if output is None:
        parser.error('指标文件为空')
    print(f'训练曲线：{output}')


if __name__ == '__main__':
    main()
