"""Pre-flight path validation for enabling HiDe (behaviour-neutral hardening).

These checks only make the existing fail-closed contract legible: the
reference installer never silently falls back, but a null path used to raise
`TypeError: expected str, bytes or os.PathLike object, not NoneType` far from
the cause. Algorithm, gates and thresholds are unchanged.
"""
import json
from types import SimpleNamespace

import pytest
import torch

from hide_nav import NavigationProjector
from hide_integration import install_hide_from_cfg, sha256


def solver3():
    m = SimpleNamespace()
    m.level_models = [torch.nn.Linear(1, 1) for _ in range(3)]
    m.level_solvers = {i: SimpleNamespace(model=torch.nn.Module()) for i in (1, 2, 3)}
    m.hide_nav_enabled = False
    return m


def assets(tmp_path):
    ckpt = tmp_path / "model.ckpt"
    norm = tmp_path / "normalizer.pt"
    ckpt.write_bytes(b"frozen checkpoint")
    norm.write_bytes(b"frozen normalizer")
    head = tmp_path / "projector.pt"
    torch.save(
        {
            "architecture": {"latent_dim": 384, "hidden_dim": 128},
            "state_dict": NavigationProjector().state_dict(),
            "metadata": {
                "source_split": "train",
                "render_domain": "live_eval_pipeline",
                "source_data_sha256": "train_sha",
                "checkpoint_sha256": sha256(ckpt),
                "normalizer_sha256": sha256(norm),
                "real_state_gate": True,
                "goal_encoder_parity_passed": True,
            },
        },
        head,
    )
    audit = tmp_path / "prediction_audit.json"
    audit.write_text(
        json.dumps(
            {
                "status": "PASS",
                "source_split": "train_heldout",
                "head_sha256": sha256(head),
                "checkpoint_sha256": sha256(ckpt),
                "num_pairs": 64,
            }
        )
    )
    scale = tmp_path / "cost_scales.json"
    scale.write_text(
        json.dumps(
            {
                "source_split": "train",
                "calibration": "frozen_rollouts",
                "head_sha256": sha256(head),
                "checkpoint_sha256": sha256(ckpt),
                "normalizer_sha256": sha256(norm),
                "num_l2": 256,
                "num_l3": 256,
                "scale_l2": 0.3,
                "scale_l3": 0.8,
            }
        )
    )
    return ckpt, norm, head, audit, scale


def cfg_for(head, audit, scale):
    return {
        "hide_nav": {
            "enabled": True,
            "projector_path": head,
            "prediction_audit_path": audit,
            "scale_stats_path": scale,
            "alpha_l2": 0.25,
            "alpha_l3": 0.25,
        }
    }


@pytest.mark.parametrize(
    "key", ["projector_path", "prediction_audit_path", "scale_stats_path"]
)
def test_null_path_is_named_not_a_type_error(tmp_path, key):
    ckpt, norm, head, audit, scale = assets(tmp_path)
    cfg = cfg_for(head, audit, scale)
    cfg["hide_nav"][key] = None
    m = solver3()
    with pytest.raises(ValueError, match=key):
        install_hide_from_cfg(cfg, m, checkpoint=ckpt, normalizer=norm)
    assert not m.hide_nav_enabled


@pytest.mark.parametrize(
    "key", ["projector_path", "prediction_audit_path", "scale_stats_path"]
)
def test_missing_key_is_named(tmp_path, key):
    ckpt, norm, head, audit, scale = assets(tmp_path)
    cfg = cfg_for(head, audit, scale)
    del cfg["hide_nav"][key]
    with pytest.raises(ValueError, match=key):
        install_hide_from_cfg(cfg, solver3(), checkpoint=ckpt, normalizer=norm)


def test_nonexistent_file_names_the_option_and_path(tmp_path):
    ckpt, norm, head, audit, scale = assets(tmp_path)
    cfg = cfg_for(head, audit, scale)
    cfg["hide_nav"]["scale_stats_path"] = str(tmp_path / "absent.json")
    with pytest.raises(FileNotFoundError, match="scale_stats_path.*absent.json"):
        install_hide_from_cfg(cfg, solver3(), checkpoint=ckpt, normalizer=norm)


def test_valid_paths_still_install(tmp_path):
    ckpt, norm, head, audit, scale = assets(tmp_path)
    m = solver3()
    install_hide_from_cfg(cfg_for(head, audit, scale), m, checkpoint=ckpt, normalizer=norm)
    assert m.hide_nav_enabled
    assert not hasattr(m.level_solvers[1].model, "hide_nav_cost")
