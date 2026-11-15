"""Check expected predictor-space gate on synthetic, NOT scientific data."""
import numpy as np
import torch
from hide_nav import NavigationProjector
from scripts.audit_hide_predictions import evaluate_prediction_pairs


def identity_xy_head():
    h=NavigationProjector(latent_dim=384,hidden_dim=128)
    with torch.no_grad():
        for layer in (h.net[0],h.net[2],h.net[4]):
            layer.weight.zero_()
            layer.bias.zero_()
            layer.weight[0,0]=1.0
            layer.weight[1,1]=1.0
    return h


def make_arrays(n=40, prediction_delta=0.0):
    start=np.zeros((n,384),dtype=np.float32)
    end=np.zeros((n,384),dtype=np.float32)
    pred=np.zeros((n,384),dtype=np.float32)
    end[:,:2]=1.0
    pred[:,:2]=1.0+prediction_delta
    xy=np.ones((n,2),dtype=np.float32)
    return start,pred,end,xy


def test_real_predicted_pairs_pass():
    out=evaluate_prediction_pairs(identity_xy_head(),*make_arrays())
    assert out["status"]=="PASS"
    assert out["median_pred_xy_error"]==0.0
    assert out["mean_motion_direction_cosine"]>0.99


def test_false_future_navigation_is_rejected():
    out=evaluate_prediction_pairs(
        identity_xy_head(), *make_arrays(prediction_delta=5.0)
    )
    assert out["status"]=="NO-GO"
