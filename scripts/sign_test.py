"""Paired comparison of two variants from a science_batch result (one-sided sign test).
    python3 scripts/sign_test.py runs/science_batch_<time>.json B A [--battery 25]
"""
import argparse
import json
import math

ap = argparse.ArgumentParser()
ap.add_argument('path')
ap.add_argument('better', help='variant expected to be better, e.g. B or K')
ap.add_argument('base', help='baseline variant, e.g. A or B')
ap.add_argument('--battery', type=float, default=None)
a = ap.parse_args()
rows = [r for r in json.load(open(a.path)) if a.battery is None or r['battery0'] == a.battery]
key = lambda r: (r['level'], r['seed'], r['battery0'])  # noqa: E731
X = {key(r): r for r in rows if r['variant'] == a.better}
Y = {key(r): r for r in rows if r['variant'] == a.base}
pairs = [(X[k], Y[k]) for k in X if k in Y]
print(f'{len(pairs)} pairs: {a.better} vs {a.base}')
for metric, sign in (('collected', 1), ('score', 1), ('penalties', -1), ('battery', 1)):
    d = [(x[metric] - y[metric]) * sign for x, y in pairs]
    w, l = sum(v > 1e-9 for v in d), sum(v < -1e-9 for v in d)
    n = w + l
    p = sum(math.comb(n, k) for k in range(w, n + 1)) / 2 ** n if n else 1.0
    print(f'{metric:10s} {a.better} better {w:3d}, worse {l:3d}, ties {len(d) - n:3d}   one-sided p = {p:.4f}')
