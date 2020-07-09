from py_bg_bootstrap import Bootstrapper


def test_dist(poisson_array):
    parr = poisson_array
    bst = Bootstrapper(bg_imgs=parr, division=1)
    cut_offs = bst.compute_confidence(threshold=95)
    assert cut_offs[0, 0] == 5.0



