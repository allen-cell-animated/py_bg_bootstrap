from py_bg_bootstrap import Bootstrapper
from numpy import nan


def test_dist(poisson_array):
    parr = poisson_array
    bst = Bootstrapper(bg_imgs=parr, division=1)
    cut_offs = bst.compute_confidence(threshold=95)
    assert cut_offs[0, 0] == 5.0 or cut_offs[0, 0] == 4.0


def test_masked_dist(poisson_masked_array):
    parr = poisson_masked_array
    bst = Bootstrapper(bg_imgs=parr, division=1)
    cut_offs = bst.compute_confidence(threshold=95)
    assert cut_offs[0, 0] is not nan
