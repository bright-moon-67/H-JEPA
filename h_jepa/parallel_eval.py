#!/usr/bin/env python3
"""多进程并行评测（把 eval trajs 切片，每个 worker 一个独立进程/World）。

背景：`eval.py` 默认 `eval.chunk_size=1`，即 50 个任务串行跑 50 次；
`solver.solvers.level*.batch_size=1` 又让 solver 逐个环境求解，GPU 利用率很低。
本脚本两层并行都用上：
  1) 每个 worker 内部 `+eval.chunk_size=null` + `batch_size=切片大小` → 一张卡上向量化批量求解；
  2) 多个 worker 进程同时跑 → 用一个进程等环境步进/Python 开销时 GPU 的空闲时间。

随机种子按官方分块语义对齐：第 k 片（全局起点 offset）用 seed = SEED + offset，
与 `+eval.chunk_size=<片大小>` 的串行分块结果一致。

用法：
  python parallel_eval.py --config-name pusht_l2 \
      --policy $HJEPA_HOME/ckpts/pusht/pusht_hjepa_l2/seed42/pusht_hjepa_l2_object.ckpt \
      --workers 4 --batch-size 16
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import yaml

HERE = Path(__file__).resolve().parent


def detect_levels(config_name: str) -> list[str]:
    """从 config/eval/<name>.yaml 里读出 solver 有哪些 level。"""
    cfg_path = HERE / 'config' / 'eval' / f'{config_name}.yaml'
    levels = ['level1', 'level2', 'level3', 'level4']  # 兜底
    try:
        cfg = yaml.safe_load(cfg_path.read_text())
        levels = list(cfg['solver']['solvers'].keys())
    except Exception:
        pass
    return levels


def split_sizes(total: int, workers: int) -> list[int]:
    base, rem = divmod(total, workers)
    return [base + (1 if i < rem else 0) for i in range(workers)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config-name', required=True)
    ap.add_argument('--policy', required=True, help='checkpoint 路径')
    ap.add_argument('--num-tasks', type=int, default=50)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--batch-size', type=int, default=16,
                    help='每个 worker 内部 solver 的 batch_size')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--env-seed-base', type=int, default=None,
                    help='确定性环境种子基数：每个任务用 env_seed_base + 全局任务 ID；'
                         '不传则保持旧的随机 settle 行为')
    ap.add_argument('--trajs', default='assets/eval_trajs/pusht/goal_offset_75_val.pt')
    ap.add_argument('--out', default=None, help='结果根目录(相对 checkpoint 目录)')
    args = ap.parse_args()

    out = args.out or f'eval_{args.config_name}_par'
    sizes = [s for s in split_sizes(args.num_tasks, args.workers) if s > 0]
    offsets, acc = [], 0
    for s in sizes:
        offsets.append(acc)
        acc += s

    payload = torch.load(args.trajs, map_location='cpu', weights_only=False)
    total = len(payload['data'])
    if total < args.num_tasks:
        print(f'✗ {args.trajs} 只有 {total} 个任务，少于 --num-tasks={args.num_tasks}')
        return 1

    levels = detect_levels(args.config_name)
    tmpdir = Path(tempfile.mkdtemp(prefix='eval_slices_'))
    print(f'→ {args.workers} 个 worker，切片 {sizes}，batch_size={args.batch_size}，'
          f'levels={levels}')

    procs, slice_paths = [], []
    manifest_workers = []
    for k, (off, size) in enumerate(zip(offsets, sizes)):
        sl = dict(payload)
        sl['data'] = [dict(ep) for ep in payload['data'][off:off + size]]
        if payload.get('episodes_idx') is not None:
            sl['episodes_idx'] = list(payload['episodes_idx'])[off:off + size]
        sp = tmpdir / f'slice_{k}.pt'
        torch.save(sl, sp)
        slice_paths.append(sp)

        task_ids = [int(x) for x in sl['episodes_idx']] if sl.get('episodes_idx') is not None else list(range(off, off + size))
        manifest_workers.append({
            'worker': k,
            'tasks': task_ids,
            'planner_seed': args.seed + off,
            'env_seed_by_task': (
                None if args.env_seed_base is None
                else {str(t): args.env_seed_base + t for t in task_ids}
            ),
        })

        overrides = [
            f'--config-name={args.config_name}',
            f'seed={args.seed + off}',                 # 对齐官方分块种子
            f'output.dir={out}/w{k}',
            f'eval.num_eval={size}',
            '+eval.chunk_size=null',                   # 一个 World 装下整片
            f'load_eval_trajs_path={sp}',
            f'policy={args.policy}',
        ]
        if args.env_seed_base is not None:
            overrides.append(f'+eval.env_seed={args.env_seed_base}')
        overrides += [f'solver.solvers.{lv}.batch_size={args.batch_size}'
                      for lv in levels]
        log = tmpdir / f'worker_{k}.log'
        print(f'  worker {k}: tasks[{off}:{off + size}] seed={args.seed + off} → {log}')
        procs.append((k, subprocess.Popen(
            [sys.executable, 'eval.py', *overrides],
            cwd=HERE, stdout=log.open('w'), stderr=subprocess.STDOUT,
        ), log))

    failed = []
    for k, p, log in procs:
        rc = p.wait()
        if rc != 0:
            failed.append((k, rc, log))
    if failed:
        for k, rc, log in failed:
            print(f'✗ worker {k} 退出码 {rc}，日志 {log}:')
            print(log.read_text()[-2000:])
        return 1

    # ---- 聚合 ----
    ckpt_dir = Path(args.policy).resolve().parent
    succ, steps, times = [], [], []
    for k in range(len(sizes)):
        m = ckpt_dir / out / f'w{k}' / 'metrics.yaml'
        if not m.exists():
            print(f'✗ 缺少 {m}')
            return 1
        d = yaml.safe_load(m.read_text())
        succ += list(d['episode_successes'])
        steps += list(d.get('steps_to_success') or [])
        times.append(float(d.get('evaluation_time', 0.0)))

    n = len(succ)
    rate = 100.0 * sum(bool(x) for x in succ) / max(n, 1)
    print(f'\n=== 并行评测结果 ({args.config_name}, {n} 任务, {args.workers} 进程) ===')
    print(f'success_rate = {rate:.1f}%  ({sum(bool(x) for x in succ)}/{n})')
    print(f'各 worker 耗时: {[round(t, 1) for t in times]} s  '
          f'(并行墙钟≈max={max(times):.1f}s, 串行合计={sum(times):.1f}s)')
    combined = ckpt_dir / out / 'metrics_combined.yaml'
    combined.write_text(yaml.safe_dump(
        {'success_rate': rate, 'episode_successes': [bool(x) for x in succ],
         'num_tasks': n, 'workers': args.workers, 'batch_size': args.batch_size,
         'worker_seconds': times}, sort_keys=False))

    manifest = {
        'format': 'parallel_eval_manifest_v1',
        'config_name': args.config_name,
        'policy': str(Path(args.policy).resolve()),
        'trajs': str(Path(args.trajs).resolve()),
        'workers': args.workers,
        'batch_size': args.batch_size,
        'planner_seed': args.seed,
        'env_seed_base': args.env_seed_base,
        'per_worker': manifest_workers,
        'git_commit': subprocess.run(
            ['git', 'rev-parse', 'HEAD'], cwd=HERE, capture_output=True, text=True
        ).stdout.strip(),
    }
    manifest_path = ckpt_dir / out / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=1))
    print(f'合并结果 → {combined}')
    print(f'运行清单 → {manifest_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
