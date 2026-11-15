"""Install/manifest gates: missing evidence and stale files FAIL CLOSED."""
import json
from types import SimpleNamespace

import pytest
import torch

from hide_nav import NavigationProjector
from hide_integration import install_hide_from_cfg, sha256


def solver3():
    m=SimpleNamespace()
    m.level_models=[torch.nn.Linear(1,1) for _ in range(3)]
    m.level_solvers={
        i:SimpleNamespace(model=torch.nn.Module()) for i in (1,2,3)
    }
    m.hide_nav_enabled=False
    return m


def assets(tmp_path):
    ckpt=tmp_path/"model.ckpt"
    norm=tmp_path/"normalizer.pt"
    ckpt.write_bytes(b"fixed frozen HJEPA checkpoint")
    norm.write_bytes(b"fixed training normalizer")
    head=tmp_path/"projector.pt"
    h=NavigationProjector()
    torch.save({
        "architecture":{"latent_dim":384,"hidden_dim":128},
        "state_dict":h.state_dict(),
        "metadata":{
            "source_split":"train",
            "render_domain":"live_eval_pipeline",
            "source_data_sha256":"test_train_dataset_sha",
            "checkpoint_sha256":sha256(ckpt),
            "normalizer_sha256":sha256(norm),
            "real_state_gate":True,
            "goal_encoder_parity_passed":True,
        },
    },head)
    prediction=tmp_path/"prediction_audit.json"
    prediction.write_text(json.dumps({
        "status":"PASS",
        "source_split":"train_heldout",
        "head_sha256":sha256(head),
        "checkpoint_sha256":sha256(ckpt),
        "num_pairs":64
    }))
    scale=tmp_path/"cost_scales.json"
    scale.write_text(json.dumps({
        "source_split":"train",
        "calibration":"frozen_rollouts",
        "head_sha256":sha256(head),
        "checkpoint_sha256":sha256(ckpt),
        "normalizer_sha256":sha256(norm),
        "num_l2":256,"num_l3":256,
        "scale_l2":0.3,"scale_l3":0.8
    }))
    cfg={"hide_nav":{
        "enabled":True,
        "projector_path":str(head),
        "prediction_audit_path":str(prediction),
        "scale_stats_path":str(scale),
        "alpha_l2":0.25,"alpha_l3":0.25
    }}
    return ckpt,norm,cfg,prediction


def test_default_off_has_no_side_effect(tmp_path):
    m=solver3()
    install_hide_from_cfg({},m,checkpoint=tmp_path/"absent.ckpt",
                          normalizer=tmp_path/"absent.pt")
    assert not m.hide_nav_enabled
    assert not hasattr(m.level_solvers[1].model,"hide_nav_cost")


def test_enabled_installs_only_l2_l3_and_checks_hashes(tmp_path):
    ckpt,norm,cfg,_=assets(tmp_path)
    m=solver3()
    manifest=install_hide_from_cfg(cfg,m,checkpoint=ckpt,normalizer=norm)
    assert m.hide_nav_enabled
    assert manifest["checkpoint_sha256"]==sha256(ckpt)
    assert not hasattr(m.level_solvers[1].model,"hide_nav_cost")
    for lev in (2,3):
        adapter=m.level_solvers[lev].model.hide_nav_cost
        assert adapter.level==lev
        assert not any(p.requires_grad for p in adapter.projector.parameters())


def test_missing_prediction_gate_fail_closed(tmp_path):
    ckpt,norm,cfg,prediction=assets(tmp_path)
    prediction.write_text(json.dumps({"status":"NO-GO"}))
    m=solver3()
    with pytest.raises(ValueError,match="Prediction"):
        install_hide_from_cfg(cfg,m,checkpoint=ckpt,normalizer=norm)
    assert not m.hide_nav_enabled


def test_checkpoint_mismatch_fail_closed(tmp_path):
    ckpt,norm,cfg,_=assets(tmp_path)
    ckpt.write_bytes(b"unexpected changed model")
    with pytest.raises(ValueError,match="checkpoint"):
        install_hide_from_cfg(cfg,solver3(),checkpoint=ckpt,normalizer=norm)


def test_refuse_double_installation(tmp_path):
    ckpt,norm,cfg,_=assets(tmp_path)
    m=solver3()
    install_hide_from_cfg(cfg,m,checkpoint=ckpt,normalizer=norm)
    with pytest.raises(RuntimeError,match="twice"):
        install_hide_from_cfg(cfg,m,checkpoint=ckpt,normalizer=norm)
