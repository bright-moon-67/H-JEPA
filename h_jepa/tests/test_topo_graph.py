import numpy as np
import pytest
import torch
from topo_graph import ObservedTopology, build_observed_topology
from goal_adapter import AdapterConfig, TopologyGoalAdapter


def make_graph():
    # A U-shaped actual trajectory. Topological shortcut A->G is absent.
    cells=[
        (0,0),(1,0),(2,0),(2,1),(2,2),(1,2),(0,2)
    ]
    xy=np.asarray(cells,dtype=float)+0.5
    episode=np.zeros(len(xy),dtype=int)
    steps=np.arange(len(xy),dtype=int)
    g=build_observed_topology(
        xy,episode,steps,resolution=1.,
        evidence_id="train_dense_xy",checkpoint_sha256="abc",
        normalizer_sha256="def"
    )
    embeds={
        "embed_1":np.stack([xy[:,0],xy[:,1],xy[:,0]+1],axis=1),
        "embed_2":np.stack([xy[:,0],xy[:,1],xy[:,1]+1],axis=1),
        "embed_3":np.stack([xy[:,0],xy[:,1],xy[:,0]-xy[:,1]],axis=1),
    }
    g.attach_live_representatives(xy,embeds)
    return g,xy,embeds


def original():
    return {
        "embed_1":torch.tensor([[[0.,0.,1.]]]),
        "embed_2":torch.tensor([[[0.,0.,2.]]]),
        "embed_3":torch.tensor([[[0.,0.,3.]]]),
    }


def test_directed_observed_traversal_only():
    g,xy,_=make_graph()
    assert len(g.edges)==6
    s=g.locate(xy[0],radius=.3)
    t=g.locate(xy[-1],radius=.3)
    path,cost=g.shortest(s,t)
    assert len(path)==7
    assert cost==6
    assert g.shortest(t,s)[0]==()  # reverse edge not inferred


def test_no_cross_wall_bridge_even_with_nearby_cells():
    g,xy,_=make_graph()
    a=g.locate(xy[0],radius=.3)
    b=g.locate(xy[-1],radius=.3)
    assert not any(j==b for j,_,_ in g.adj[a])
    suggestion=g.suggest(xy[0],xy[-1],
        attachment_radius=.3,attachment_margin=0.,lookahead_cells=2)
    assert suggestion is not None
    assert suggestion.waypoint_xy==(2.5,0.5)


def test_two_trajectories_can_share_same_physical_cell():
    xy=np.array([[.5,.5],[1.5,.5],[2.5,.5],
                 [2.5,.5],[2.5,1.5],[2.5,2.5]],float)
    ep=np.array([0,0,0,1,1,1])
    step=np.array([0,1,2,0,1,2])
    g=build_observed_topology(xy,ep,step,resolution=1.)
    a=g.locate(xy[0],radius=.2)
    b=g.locate(xy[-1],radius=.2)
    path,cost=g.shortest(a,b)
    assert len(path)==5 and cost==4


def test_true_nonlocal_jump_discarded():
    g=build_observed_topology(
        np.array([[.5,.5],[3.5,.5]]),np.array([0,0]),
        np.array([0,1]),resolution=1.0
    )
    assert not len(g.edges)
    assert g.discarded_nonlocal_transitions==1


def test_sparse_sampled_gap_never_creates_a_shortcut():
    g=build_observed_topology(
        np.array([[.5,.5],[1.5,.5]]),np.array([0,0]),
        np.array([0,10]),resolution=1.0
    )
    assert len(g.edges)==0


def test_no_val_or_official_data_in_graph():
    with pytest.raises(ValueError,match="TRAIN"):
        build_observed_topology(
            np.array([[.5,.5],[1.5,.5]]),np.array([0,0]),
            np.array([0,1]),resolution=1.,source_split="val"
        )


def test_checkpoint_normalizer_provenance():
    g,_,_=make_graph()
    g.verify("abc","def")
    with pytest.raises(ValueError,match="mismatch"):
        g.verify("abc","incorrect")


def test_disabled_exact_identity_and_no_rng_consumption():
    g,xy,_=make_graph()
    ctrl=TopologyGoalAdapter(g,AdapterConfig(enabled=False))
    orig=original()
    out,mask=ctrl.choose(xy[:1],xy[-1:],orig,planning_call=0)
    assert out is orig and not mask.any()


def test_waypoint_changes_all_three_goal_levels_consistently():
    g,xy,embeds=make_graph()
    ctrl=TopologyGoalAdapter(
        g,AdapterConfig(enabled=True,certification_evidence_id="train-local-rollouts",checkpoint_sha256="abc",normalizer_sha256="def",attach_radius=.3,
                        lookahead_cells=2,arrival_radius=.3,
                        max_waypoint_calls=3)
    )
    orig=original()
    out,mask=ctrl.choose(xy[:1],xy[-1:],orig,planning_call=0)
    assert mask.tolist()==[True]
    assert torch.equal(orig["embed_3"],original()["embed_3"])
    for key in ("embed_1","embed_2","embed_3"):
        torch.testing.assert_close(out[key][0,0],torch.tensor(
            embeds[key][2],dtype=torch.float32))
    # Once actual position reaches the first waypoint, move to the next.
    out2,mask2=ctrl.choose(xy[2:3],xy[-1:],orig,planning_call=1)
    assert mask2.tolist()==[True]
    torch.testing.assert_close(out2["embed_3"][0,0],torch.tensor(
        embeds["embed_3"][4],dtype=torch.float32))


def test_timeout_and_no_usable_route_fallback():
    g,xy,_=make_graph()
    ctrl=TopologyGoalAdapter(
        g,AdapterConfig(enabled=True,certification_evidence_id="train-local-rollouts",checkpoint_sha256="abc",normalizer_sha256="def",attach_radius=.3,
                        arrival_radius=.2,max_waypoint_calls=2)
    )
    orig=original()
    out,mask=ctrl.choose(xy[:1],xy[-1:],orig,planning_call=0)
    assert mask.item()
    out,mask=ctrl.choose(xy[:1],xy[-1:],orig,planning_call=1)
    assert mask.item()
    out,mask=ctrl.choose(xy[:1],xy[-1:],orig,planning_call=2)
    assert not mask.item()
    assert torch.equal(out["embed_3"],orig["embed_3"])

    ctrl.reset(1)
    out,mask=ctrl.choose(
        np.array([[999.,999.]]),xy[-1:],orig,planning_call=0
    )
    assert not mask.item()
    assert torch.equal(out["embed_1"],orig["embed_1"])


def test_ambiguous_position_is_rejected():
    g,xy,_=make_graph()
    # exactly on boundary between adjacent cell centers
    assert g.locate(np.array([1.,.5]),radius=1.,margin=.1) is None


def test_goal_adapter_rejects_shape_mismatch():
    g,xy,_=make_graph()
    ctrl=TopologyGoalAdapter(g,AdapterConfig(enabled=True,certification_evidence_id="train-local-rollouts",checkpoint_sha256="abc",normalizer_sha256="def"))
    with pytest.raises(ValueError,match="Missing"):
        ctrl.choose(xy[:1],xy[-1:],
            {"embed_3":torch.zeros(1,1,3)},planning_call=0)


def test_roundtrip_map_without_pickle(tmp_path):
    from topo_graph import save_topology,load_topology
    g,xy,_=make_graph()
    fn=tmp_path/"train_topology.npz"
    save_topology(g,str(fn))
    loaded=load_topology(str(fn))
    assert loaded.diagnostic()==g.diagnostic()
    loaded.verify("abc","def")
    out=loaded.suggest(xy[0],xy[-1],attachment_radius=.3,
        attachment_margin=0.,lookahead_cells=2)
    assert out.waypoint_node in loaded.reps


def test_explicit_certification_required():
    g,_,_=make_graph()
    with pytest.raises(ValueError,match="Uncertified"):
        TopologyGoalAdapter(g,AdapterConfig(enabled=True))
