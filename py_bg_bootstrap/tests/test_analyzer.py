import pytest
from ..analyzer import Analyzer


def test_poisson_confidence(poisson_array):
    ana = Analyzer(bg_imgs=poisson_array)
    assert ana.compute_confidence(threshold=95.0) == 5.0


def test_gaussian_confidence(gaussian_array):
    ana = Analyzer(bg_imgs=gaussian_array)
    assert ana.compute_confidence(threshold=95.0) == pytest.approx(1.6, 0.1)


def test_shapiro_gaussian(gaussian_array):
    ana = Analyzer(bg_imgs=gaussian_array)
    assert ana.shapiro()


def test_not_shapiro_gaussian(poisson_array):
    ana = Analyzer(bg_imgs=poisson_array)
    assert not ana.shapiro()


def test_anderson_gaussian(gaussian_array):
    ana = Analyzer(bg_imgs=gaussian_array)
    assert ana.anderson_darling()


def test_not_anderson_gaussian(poisson_array):
    ana = Analyzer(bg_imgs=poisson_array)
    assert not ana.anderson_darling()
