"""Stage-1 certification must fail closed on ambiguous local reach data."""
import pytest
from scripts.calibrate_pahrl_attachments import calibrate

def rows(ok=True):
    positives = [0.01, 0.02, 0.03, 0.04]
    negatives = [0.10, 0.11, 0.12, 0.13] if ok else [0.02, 0.10, 0.11, 0.12]
    return [
        {"cosine_distance":d, "local_success":True, "source_split":"train",
         "source_episode_uid":f"a{i}"} for i,d in enumerate(positives)
    ]+[
        {"cosine_distance":d, "local_success":False, "source_split":"train",
         "source_episode_uid":f"b{i}"} for i,d in enumerate(negatives)
    ]

def test_certification_is_conservative_single_rule():
    out=calibrate(rows())
    assert out["calibration_passed"]
    assert out["source_split"]=="train"
    assert 0.04 < out["max_cosine_distance"] < 0.10

def test_overlapping_negative_is_not_accepted():
    with pytest.raises(ValueError, match="NOT separable"):
        calibrate(rows(ok=False))

def test_validation_data_cannot_certify():
    r=rows()
    r[0]["source_split"]="val"
    with pytest.raises(ValueError, match="train-only"):
        calibrate(r)
