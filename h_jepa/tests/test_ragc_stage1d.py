"""Tests for the Stage-1D fixes.

Covered:
  1. `replay_gate` — the hidden-physics fingerprint is a HARD validity condition
     (a 0.6-unit anchor drift must invalidate the anchor even when the success
     flag happens to match).
  2. Per-pair success labels — the second antithetic pair must use candidate
     indices [2], [3], not reuse [0], [1].
"""
import numpy as np
import pytest

from ragc_collect import replay_gate


# --------------------------------------------------------------------------- #
# 1. replay gate
# --------------------------------------------------------------------------- #
def test_replay_gate_accepts_exact_replay():
    ok, reason = replay_gate(fp_maxdiff=0.0, div=-1,
                             success_equal=True, rgb_equal=True)
    assert ok and reason == ""


def test_replay_gate_rejects_fingerprint_drift_even_if_success_matches():
    # 0.62 mirrors the ep7 case: identical outcome, drifted hidden state.
    ok, reason = replay_gate(fp_maxdiff=6.238e-01, div=-1,
                             success_equal=True, rgb_equal=True)
    assert not ok
    assert "fp_maxdiff" in reason


def test_replay_gate_rejects_action_divergence():
    ok, reason = replay_gate(fp_maxdiff=0.0, div=10,
                             success_equal=True, rgb_equal=True)
    assert not ok and "step 10" in reason


def test_replay_gate_rejects_rgb_and_success_mismatch():
    ok, _ = replay_gate(fp_maxdiff=0.0, div=-1, success_equal=True,
                        rgb_equal=False)
    assert not ok
    ok, _ = replay_gate(fp_maxdiff=0.0, div=-1, success_equal=False,
                        rgb_equal=True)
    assert not ok


def test_replay_gate_rejects_nonfinite_fingerprint():
    ok, _ = replay_gate(fp_maxdiff=float("nan"), div=-1,
                        success_equal=True, rgb_equal=True)
    assert not ok


def test_replay_gate_threshold_is_one_e_minus_seven():
    ok, _ = replay_gate(fp_maxdiff=1e-8, div=-1,
                        success_equal=True, rgb_equal=True)
    assert ok
    ok, _ = replay_gate(fp_maxdiff=1.1e-7, div=-1,
                        success_equal=True, rgb_equal=True)
    assert not ok


# --------------------------------------------------------------------------- #
# 2. per-pair success indexing
# --------------------------------------------------------------------------- #
def _fake_anchor_record():
    """Two directions => 4 candidates; give each a DIFFERENT success pattern."""
    return {
        "kind": "anchor", "split": "train", "valid": True,
        "anchor_id": 0, "task_uid": "t", "source_episode_uid": "e",
        "baseline_success": False, "astar_cost": 1.0,
        "z0": np.zeros(192, np.float32),
        "goal_emb": np.zeros(192, np.float32),
        "goal_is_upper": False, "epsilon": 0.3,
        "astar_actions": np.zeros((2, 10), np.float32),
        "pairs": [
            {"plus": np.zeros((2, 10), np.float32),
             "minus": np.zeros((2, 10), np.float32),
             "plus_cost": 1.0, "minus_cost": 0.5},
            {"plus": np.zeros((2, 10), np.float32),
             "minus": np.zeros((2, 10), np.float32),
             "plus_cost": 2.0, "minus_cost": 1.5},
        ],
        "candidate_successes": [True, False, False, True],
        "candidate_costs": [1.0, 0.5, 2.0, 1.5],
    }


def test_second_pair_uses_its_own_success_labels(tmp_path):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "stage1_report",
        Path(__file__).resolve().parents[1] / "scripts" / "ragc_stage1_report.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    made = mod.rebuild_npz([_fake_anchor_record()], str(tmp_path))
    ds = made["train"][1]
    # pair 0 -> candidates 0,1 ; pair 1 -> candidates 2,3
    assert ds["plus_success"].tolist() == [True, False]
    assert ds["minus_success"].tolist() == [False, True]
    assert ds["plus_cost"].tolist() == pytest.approx([1.0, 2.0])
    assert ds["minus_cost"].tolist() == pytest.approx([0.5, 1.5])
