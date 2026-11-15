import pytest
from scripts.compare_hide_dev import compare

def data(succ, start=1000):
    return {
        "episode_successes":succ,
        "seeds":[start+i for i in range(12)],
        "_compute_seconds":10.0,
    }

def test_paired_rescue_destroy_and_net():
    base=data([True]*8+[False]*4)
    alt=data([True]*7+[False]+[True]*3+[False])
    got=compare(base,alt)
    assert got["baseline_successes"]==8
    assert got["hide_successes"]==10
    assert got["net"]==2
    assert got["rescue"]==3
    assert got["destroy"]==1

def test_reject_seed_mismatch():
    a=data([False]*12,1000)
    b=data([True]*12,2000)
    with pytest.raises(ValueError,match="Not paired"):
        compare(a,b)
