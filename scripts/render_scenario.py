"""Render the 4-frame scenario visualisation from a real run (judge + agent logs).
    cd src/did_agent && PYTHONPATH=. python3 ../../scripts/render_scenario.py [judge.json] [agent.json] [out_dir]
Frames: 1 start & map, 2 task (LLM plan), 3 motion (replanning), 4 result.
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, Polygon  # noqa: E402

from did_agent.core.costmap import CostMap  # noqa: E402
from did_agent.core.grid_map import OCCUPIED, GridMap  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
judge_path = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob(os.path.join(ROOT, 'runs', '*_judge_medium*.json')))[-1]
agent_path = sys.argv[2] if len(sys.argv) > 2 else sorted(glob.glob(os.path.join(ROOT, 'runs', '*_agent.json')))[-1]
out_dir = sys.argv[3] if len(sys.argv) > 3 else os.path.join(ROOT, 'docs', 'img')
J, A = json.load(open(judge_path)), json.load(open(agent_path))
sc = J['scenario']
traj = np.array([[p[0], p[1], p[2]] for p in J['trajectory']])
journal = A['journal']
llm = A.get('llm_journal', [])

# palette = the control panel's light theme
C = dict(bg='#ffffff', free='#ffffff', margin='#fbe3e6', obstacle='#3f3f46', path='#0284c7', sample='#eab308', done='#a1a1aa',
         terrain='#d97706', base='#16a34a', trail='#d97706', text='#18181b', muted='#71717a')

grid = GridMap.from_yaml(os.path.join(ROOT, 'src', 'did_agent', 'maps', 'map.yaml'))
cm = CostMap(grid, inflation_radius=0.2)
img = np.ones(grid.occ.shape + (4,))
img[..., 3] = 0.0                                                    # unknown = transparent
for mask, col in ((grid.occ == 0, C['free']), (cm.lethal & (grid.occ == 0), C['margin']), (grid.occ == OCCUPIED, C['obstacle'])):
    rgb = [int(col[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    img[mask] = rgb + [1.0]
extent = (grid.origin_x, grid.origin_x + grid.width * grid.resolution, grid.origin_y, grid.origin_y + grid.height * grid.resolution)
base = sc['base']
samples = {s['id']: (s['x'], s['y']) for s in sc['samples']}
collect_t = {}
for j in journal:
    if j['kind'] == 'collect' and j['success']:
        sid = j['message'].split()[1]
        collect_t[sid] = j['t']


def frame(ax, upto, title, route=None, note=None, show_trail=True, robot_t=None):
    ax.set_facecolor(C['bg'])
    ax.imshow(img, extent=extent, origin='lower', interpolation='nearest', zorder=0)
    for z in sc['terrain']:
        ax.add_patch(Circle((z['cx'], z['cy']), z['r'], facecolor=C['terrain'], alpha=.16, edgecolor=C['terrain'], lw=1.2, zorder=1))
        ax.text(z['cx'], z['cy'], f"{z['id']} ×{str(z['multiplier']).replace('.', ',')}", color=C['terrain'], ha='center', va='center', fontsize=9, fontweight='bold', zorder=2)
    ax.add_patch(Circle(base, 0.3, facecolor=C['base'], alpha=.15, edgecolor=C['base'], lw=1.2, zorder=1))
    ax.text(base[0], base[1] - 0.45, 'база', color=C['base'], ha='center', fontsize=9, fontweight='bold')
    if route:
        pts = [base] + [samples[s] for s in route] + [base]
        for k, (a, b) in enumerate(zip(pts, pts[1:])):
            ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=11, color=C['path'], lw=1.6, linestyle=(0, (4, 3)), shrinkA=8, shrinkB=8, zorder=3))
        for k, s in enumerate(route):
            x, y = samples[s]
            ax.text(x + 0.13, y + 0.13, str(k + 1), color='white', fontsize=8, fontweight='bold', ha='center', va='center', zorder=6,
                    bbox=dict(boxstyle='circle,pad=0.25', fc=C['path'], ec='none'))
    if show_trail and upto > 0:
        tr = traj[traj[:, 0] <= upto]
        ax.plot(tr[:, 1], tr[:, 2], color=C['trail'], lw=2, alpha=.9, zorder=4, solid_capstyle='round')
    for sid, (x, y) in samples.items():
        done = sid in collect_t and collect_t[sid] <= upto
        ax.add_patch(Circle((x, y), 0.07, facecolor=C['done'] if done else C['sample'], edgecolor='white', lw=1.5, zorder=5))
        ax.text(x, y - 0.2, sid + (' ✓' if done else ''), color=C['muted'] if done else C['text'], ha='center', fontsize=8.5, fontweight='bold', zorder=5)
    t_r = upto if robot_t is None else robot_t
    k = max(0, np.searchsorted(traj[:, 0], t_r) - 1)
    rx, ry = traj[k, 1], traj[k, 2]
    if k + 1 < len(traj):
        dx, dy = traj[min(k + 2, len(traj) - 1), 1] - rx, traj[min(k + 2, len(traj) - 1), 2] - ry
    else:
        dx, dy = 1.0, 0.0
    yaw = np.arctan2(dy, dx) if abs(dx) + abs(dy) > 1e-3 else 0.0
    ax.add_patch(Circle((rx, ry), 0.11, facecolor=C['path'], alpha=.2, edgecolor=C['path'], lw=1.8, zorder=7))
    tri = np.array([[0.17, 0], [-0.05, 0.08], [-0.05, -0.08]])
    rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    ax.add_patch(Polygon(tri @ rot.T + [rx, ry], closed=True, facecolor=C['path'], zorder=8))
    if note:
        ax.text(0.5, 0.0, note, transform=ax.transAxes, ha='center', va='bottom', fontsize=8.5, wrap=True, color=C['text'],
                bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#e4e4e7'))
    ax.set_title(title, loc='left', fontsize=12, fontweight='bold', color=C['text'], pad=8)
    ax.set_xlim(-3.05, 3.05)
    ax.set_ylim(-2.85, 2.85)
    ax.set_aspect('equal')
    ax.axis('off')


plan1 = [s.split()[1] for s in llm[0]['plan'] if s.startswith('goto')] if llm else list(samples)
t_first = next(j['t'] for j in journal if j['kind'] == 'collect')
t_mid = sorted(collect_t.values())[min(2, len(collect_t) - 1)]
t_end = traj[-1, 0]
replan = next((r for r in llm[1:] if r.get('trigger')), None)
frames = [
    ('1. Запуск: карта и робот на базе', 0, None, 'карта стоимостей: розовое — запретная зона 0,2 м', False, 0),
    ('2. Задача: план от LLM', 0, plan1, 'LLM: ' + ' → '.join(plan1 + ['база']), False, 0),
    ('3. Движение и перепланирование', t_mid, None,
     ('дорогой переезд → новый план: ' + ' → '.join([s.split()[1] for s in replan['plan'] if s.startswith('goto')] + ['база'])) if replan else 'робот едет по пути A*', True, None),
    ('4. Результат', t_end, None,
     f"{J['score']['collected']}/{J['score']['samples_total']} образцов · возврат · штрафов {sum(J['score']['penalties'].values())} · путь {J['score']['distance']:.1f} м".replace('.', ','),
     True, None),
]
os.makedirs(out_dir, exist_ok=True)
for i, (title, upto, route, note, trail, robot_t) in enumerate(frames, 1):
    fig = plt.figure(figsize=(4.6, 4.4), dpi=200)
    fig.patch.set_facecolor('white')
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])              # map only: titles and captions live on the slide
    frame(ax, upto, '', route, None, trail, robot_t)
    path = os.path.join(out_dir, f'scenario_{i}.png')
    fig.savefig(path, facecolor='white')
    plt.close(fig)
    print(path)
print('replan trigger:', replan['trigger'] if replan else None)
