import json
from pathlib import Path

import numpy as np
import pytest
import torch

from pahrl_memory import build_graph, GraphMemory
from pahrl_waypoints import GraphWaypointController, WaypointConfig


def unit_eye(n=5):
    return np.eye(n, dtype=np.float32)


def graph(cert=False):
    evidence = (
        {"calibration_passed": True, "source_split": "train", "max_cosine_distance": 0.05,
         "num_trials": 12, "positive_trials": 6, "negative_trials": 6,
         "evidence_sha256": "hash-test", "evidence_id": "held-out-local-controller-12"}
        if cert else None
    )
    z = unit_eye()
    return build_graph(
        z,
        episode=np.zeros(5, dtype=int),
        steps=np.arange(5, dtype=int)*10,
        dataset_name="visual_antmaze_medium_explore_stitch_train",
        checkpoint_sha256="a"*64,
        certification=evidence,
    )


def test_only_observed_directed_edges():
    g = graph()
    assert np.array_equal(g.indices, np.array([1, 2, 3, 4]))
    assert g.shortest_path(0, 4)[1] == (0, 1, 2, 3, 4)
    assert g.shortest_path(4, 0)[1] == ()


def test_no_cross_episode_shortcut():
    z = unit_eye(6)
    g = build_graph(
        z, np.array([0, 0, 0, 1, 1, 1]),
        np.array([0,10,20,0,10,20]),
        checkpoint_sha256="a"*64, dataset_name="maze_train"
    )
    assert len(g.indices) == 4
    assert g.shortest_path(0, 5)[1] == ()
    assert g.shortest_path(2, 3)[1] == ()


def test_refuse_unproven_cross_episode_edge():
    with pytest.raises(ValueError, match="Unproven"):
        build_graph(
            unit_eye(3), np.array([0,0,1]), np.array([0,10,0]),
            checkpoint_sha256="a"*64,
            dataset_name="maze_train",
            validated_links=[
                {"from_node":1,"to_node":2,"successes":1,"trials":3,"mean_steps":10,"source_split":"train","evidence_sha256":"x"}
            ]
        )


def test_validated_cross_episode_edge():
    g = build_graph(
        unit_eye(4), np.array([0,0,1,1]), np.array([0,10,0,10]),
        checkpoint_sha256="a"*64,
        dataset_name="maze_train",
        validated_links=[
            {"from_node":1,"to_node":2,"successes":3,"trials":3,"mean_steps":9,"source_split":"train","evidence_sha256":"x"}
        ],
    )
    assert g.shortest_path(0,3)[1] == (0,1,2,3)


def test_online_attachment_requires_controller_calibration():
    g = graph()
    assert g.route(g.z[0], g.z[4]) is None
    with pytest.raises(ValueError, match="blocked"):
        GraphWaypointController(g, WaypointConfig(enabled=True))


def test_graph_roundtrip_checkpoint_and_validation(tmp_path):
    g = graph(cert=True)
    p = tmp_path/"graph.npz"
    g.save(p)
    reloaded = GraphMemory.load(p)
    reloaded.verify_checkpoint("a"*64)
    with pytest.raises(ValueError, match="do not match"):
        reloaded.verify_checkpoint("b"*64)
    assert reloaded.route(reloaded.z[0],reloaded.z[4]).nodes == (0,1,2,3,4)


def test_online_route_advances_only_on_reaching_state():
    g = graph(cert=True)
    ctl=GraphWaypointController(
        g, WaypointConfig(enabled=True, reach_cosine_distance=0.05,
                          k_attach=1, max_waypoint_calls=3)
    )
    ctl.reset(1)
    def query(current, t):
        return ctl.target_embeddings(
            torch.tensor(current).view(1,1,-1),
            torch.tensor(g.z[4]).view(1,1,-1),
            steps_taken=t, call_idx=t//10
        )
    out,mask=query(g.z[0],0)
    assert mask.tolist() == [True]
    torch.testing.assert_close(out[0,0],torch.tensor(g.z[1]))
    out,mask=query(g.z[1],10)
    assert mask.tolist() == [True]
    torch.testing.assert_close(out[0,0],torch.tensor(g.z[2]))
    out,mask=query(g.z[2],20)
    assert mask.tolist() == [True]
    torch.testing.assert_close(out[0,0],torch.tensor(g.z[3]))
    out,mask=query(g.z[3],30)
    assert mask.tolist() == [False]
    torch.testing.assert_close(out[0,0],torch.tensor(g.z[4]))


def test_waypoint_timeout_falls_back_to_original():
    g=graph(cert=True)
    ctl=GraphWaypointController(
        g, WaypointConfig(enabled=True, reach_cosine_distance=.05,
                          k_attach=1, max_waypoint_calls=2)
    )
    ctl.reset(1)
    for t in (0,10):
        z, m=ctl.target_embeddings(
            torch.tensor(g.z[0]).view(1,1,-1),
            torch.tensor(g.z[4]).view(1,1,-1),
            steps_taken=t, call_idx=t//10
        )
        assert m.item()
    z,mask=ctl.target_embeddings(
        torch.tensor(g.z[0]).view(1,1,-1),
        torch.tensor(g.z[4]).view(1,1,-1),
        steps_taken=20, call_idx=2
    )
    assert not mask.item()
    torch.testing.assert_close(z[0,0],torch.tensor(g.z[4]))


def test_disabled_is_exact_goal_noop():
    g=graph(cert=False)
    ctl=GraphWaypointController(g, WaypointConfig(enabled=False))
    orig=torch.tensor(g.z[4]).view(1,1,-1)
    out, mask=ctl.target_embeddings(
        torch.tensor(g.z[0]).view(1,1,-1),orig,
        steps_taken=0,call_idx=0
    )
    assert torch.equal(out,orig)
    assert not mask.any()


def test_no_graph_route_is_noop():
    z=unit_eye(4)
    cert={"calibration_passed":True,"source_split":"train","max_cosine_distance":0.1,
          "num_trials":8,"positive_trials":4,"negative_trials":4,
          "evidence_sha256":"hash-test","evidence_id":"local"}
    g=build_graph(
        z,np.array([0,0,1,1]),np.array([0,10,0,10]),
        checkpoint_sha256="x",dataset_name="maze_train",
        certification=cert
    )
    ctl=GraphWaypointController(g,WaypointConfig(enabled=True,k_attach=1))
    out,mask=ctl.target_embeddings(
        torch.tensor(z[0]).view(1,1,-1),
        torch.tensor(z[3]).view(1,1,-1),
        steps_taken=0,call_idx=0
    )
    assert not mask.any()


def test_reject_misaligned_or_val_data():
    with pytest.raises(ValueError, match="train"):
        build_graph(
            unit_eye(2), np.array([0,0]),np.array([0,10]),
            dataset_name="maze_val",checkpoint_sha256="a"
        )
    with pytest.raises(ValueError, match="sorted"):
        build_graph(
            unit_eye(2), np.array([0,0]),np.array([10,0]),
            dataset_name="maze_train",checkpoint_sha256="a"
        )
