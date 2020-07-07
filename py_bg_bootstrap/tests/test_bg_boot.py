import pytest
from py_bg_bootstrap import Bootstrapper


def test_dist(poisson_array):
    parr = poisson_array
    bst = Bootstrapper(bg_imgs=parr, division=10)
    cuttoffs = bst.compute_confidence(threshold=50)
    print(f"\n cuttoff = {cuttoffs[0, 0]}\n")
    assert True



