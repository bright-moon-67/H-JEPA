"""Calibrate the adaptive-replanning threshold ``tau`` from a ``log_only`` run.

Reads the JSONL produced by ``AdaptiveWorldModelPolicy`` (one record per
5-step check per environment) and reports:

  * the latent scale ``sigma_z^2`` (mean of ``z_sq``), used to normalise
    ``e_t = ||z_hat - z||^2 / (D sigma_z^2)``;
  * the empirical quantiles of ``e_t``;
  * the trigger rate and the extra planner calls each candidate tau implies.

Usage:
    python scripts/calibrate_adaptive.py $HJEPA_HOME/adaptive_logs/pusht_l2_seed42.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_records(path: Path) -> list[dict]:
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    if not lines:
        raise SystemExit(f'{path} is empty - did the eval actually run?')
    return [json.loads(line) for line in lines]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path, help='JSONL written by the adaptive policy')
    parser.add_argument(
        '--quantiles',
        default='50,80,90,95,99',
        help='comma separated quantiles to evaluate as candidate tau',
    )
    parser.add_argument(
        '--baseline-plans',
        type=float,
        default=9.0,
        help='planner calls per episode without adaptation (eval_budget / 10)',
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=50,
        help='number of evaluated tasks, used to derive checks per episode',
    )
    args = parser.parse_args()

    records = load_records(args.log)
    mse = np.array([r['mse'] for r in records], dtype=np.float64)
    z_sq = np.array([r['z_sq'] for r in records], dtype=np.float64)

    sigma2 = float(np.mean(z_sq))
    e = mse / (sigma2 + 1e-8)
    checks_per_episode = len(records) / max(args.episodes, 1)

    print(f'log            : {args.log}')
    print(f'checks         : {len(records)}')
    print(f'latent dim D   : {sorted({int(r["D"]) for r in records})}')
    print(f'sigma_z^2      : {sigma2:.6f}  (mean over checks of mean(z^2))')
    print(f'e_t  mean/std  : {e.mean():.6f} / {e.std():.6f}')
    print(f'e_t  min/max   : {e.min():.6f} / {e.max():.6f}')
    print()

    qs = [float(q) for q in args.quantiles.split(',')]
    values = np.percentile(e, qs)
    print(f'{"tau":>10}  {"quantile":>9}  {"trigger%":>9}  {"extra plans/ep":>15}  {"compute":>8}')
    print('-' * 72)
    for q, tau in zip(qs, values):
        rate = float((e > tau).mean())
        extra = rate * checks_per_episode
        compute = (args.baseline_plans + extra) / args.baseline_plans
        print(f'{tau:10.4f}  {q:9.0f}  {rate * 100:9.2f}  {extra:15.2f}  {compute:7.2f}x')

    print()
    print('suggested config:')
    print(f'  sigma_z: {sigma2:.6f}')
    print(f'  tau: {np.percentile(e, 90):.4f}   # P90, ~10% of checks re-plan early')


if __name__ == '__main__':
    main()
