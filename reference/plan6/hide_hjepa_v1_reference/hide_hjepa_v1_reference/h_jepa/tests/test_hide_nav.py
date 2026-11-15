"""CPU tests for active HiDe loss + strict disabled/no-op invariants."""
import numpy as np
import pytest
import torch

from hide_nav import (
    NavigationProjector, NavigationCostAdapter, navigation_match_cost,
    prepare_frozen_projector
)
from scripts.train_hide_nav import dataset_splits
from scripts.calibrate_hide_scales import scale_costs


def make_head():
    torch.manual_seed(42)
    return NavigationProjector(384, 128)


def make_inputs(b=2,n=3,t=4,g=1):
    torch.manual_seed(6)
    z = torch.randn(b,n,t,384,requires_grad=True)
    goal = torch.randn(b,n,g,384)
    return z,goal


def test_head_returns_xy_and_shape():
    h=make_head()
    z=torch.randn(2,3,4,384)
    assert h(z).shape==(2,3,4,2)


def test_shape_and_last_time_index():
    h=make_head()
    z,goal=make_inputs()
    result=navigation_match_cost(h,z,goal,cost_last_n=1)
    assert result.shape==(2,3)
    # Earlier time indices are ignored with cost_last_n=1.
    altered=z.detach().clone().requires_grad_(True)
    altered.data[:,:,:-1]+=100
    torch.testing.assert_close(
        navigation_match_cost(h,altered,goal,cost_last_n=1),
        result
    )


def test_cost_two_goals_sum_suffix():
    h=make_head()
    z,goal=make_inputs(g=2)
    x=navigation_match_cost(h,z,goal,cost_last_n=2)
    assert x.shape==(2,3)


def test_error_on_bad_broadcast_not_silently_shared():
    h=make_head()
    z,goal=make_inputs()
    with pytest.raises(ValueError,match="mismatch"):
        navigation_match_cost(h,z,goal[:,:1],cost_last_n=1)


def test_disabled_adapter_exact_identity():
    h=make_head()
    adapter=NavigationCostAdapter(h,level=3,alpha=.25,scale=1)
    z,goal=make_inputs()
    base=torch.rand(2,3)
    out=adapter(base,z,{"goal_embed_0":goal,"_hide_nav_active":False})
    assert out is base
    assert torch.equal(out,base)


def test_gradient_through_frozen_head_to_actions():
    h=prepare_frozen_projector(make_head(),"cpu")
    adapter=NavigationCostAdapter(h,level=3,alpha=.25,scale=3.)
    actions=torch.randn(2,3,4,384,requires_grad=True)
    target=torch.zeros(2,3,1,384)
    out=adapter(torch.ones(2,3),actions,{
        "goal_embed_0":target,
        "_hide_nav_active":True,
        "goal_embed_0_level":3,
        "goal_embed_0_in_upper_space":False,
        "cost_last_n":1,
    })
    assert out.shape==(2,3)
    out.sum().backward()
    assert actions.grad is not None
    assert torch.isfinite(actions.grad).all()
    assert actions.grad.abs().sum().item()>0
    assert not any(p.grad is not None for p in h.parameters())


def test_l2_upper_projected_goal():
    h=make_head()
    adapter=NavigationCostAdapter(h,level=2,alpha=.25,scale=0.5)
    z,goal=make_inputs()
    cost=adapter(torch.zeros(2,3),z,{
        "_hide_nav_active":True,
        "goal_embed_0":goal,
        "goal_embed_0_in_upper_space":True,
        "cost_last_n":1,
    })
    assert (cost>=0).all() and cost.shape==(2,3)


def test_l2_cannot_add_cost_in_own_latent_space():
    h=make_head()
    adapter=NavigationCostAdapter(h,level=2,alpha=.25,scale=1.)
    z,goal=make_inputs()
    with pytest.raises(ValueError,match="goal level"):
        adapter(torch.zeros(2,3),z,{
            "_hide_nav_active":True, "goal_embed_0":goal,
            "goal_embed_0_level":2
        })


def test_l1_never_eligible():
    with pytest.raises(ValueError,match="Only L3"):
        NavigationCostAdapter(make_head(),level=1,alpha=.25,scale=1.)


def test_split_by_episode_not_frames():
    episode=np.repeat(np.arange(30),4)
    (tr,va,te),splits=dataset_splits(episode,seed=42)
    assert set(episode[tr]).isdisjoint(set(episode[va]))
    assert set(episode[tr]).isdisjoint(set(episode[te]))
    assert set(episode[va]).isdisjoint(set(episode[te]))
    assert len(tr)+len(va)+len(te)==len(episode)


def test_scale_calibration():
    assert scale_costs(np.array([2.]*64),np.array([4.]*64))==.5
    with pytest.raises(ValueError,match="Degenerate"):
        scale_costs(np.zeros(64),np.ones(64))
    with pytest.raises(ValueError,match="32"):
        scale_costs(np.ones(4),np.ones(4))


def test_deterministic_head_loading(tmp_path):
    h=make_head()
    file=tmp_path/"head.pt"
    torch.save(h.state_dict(),file)
    h2=make_head()
    h2.load_state_dict(torch.load(file,weights_only=True))
    x=torch.randn(5,384)
    torch.testing.assert_close(h(x),h2(x))
