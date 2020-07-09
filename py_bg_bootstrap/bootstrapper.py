from math import floor

import numpy as np


class Bootstrapper(object):
    def __init__(self, bg_imgs: np.ndarray, division: int = 1):
        """
        This class is a starting point for bootstrap methods. In it's current form
        it simply sampling the data. The idea is it could be developed into a
        hypothesis testing framework.

        Parameters
        ----------
        bg_imgs a list or ndarray of 2D Images of blanks to sample from
        division (optional) number of regions to chop each dimension into.
        For example divisions=3 would result in 9 independently handled patches.
        The default is 1.
        """
        if not Bootstrapper._consistent_shape(bg_imgs):
            raise ValueError("Bootstrapper: bg_imgs have inconsistent shape / "
                             "different sizes.")
        if len(bg_imgs.shape) != 3:
            raise ValueError("Bootstrapper: bg_images must be 3D. (D, Y, X) where D is"
                             " index or Z or T.")
        self._bg_images = bg_imgs
        self._divisions = division
        if self._divisions < 1:
            self._divisions = 1
        self._x_start_stop, self._y_start_stop = self.compute_grid()
        self._samples = 10000
        self._rng = np.random.default_rng()  # setup random number generator

    def compute_confidence(self, threshold: int):
        x_size = len(self._x_start_stop)
        y_size = len(self._y_start_stop)
        z_size = self._bg_images.shape[0]
        thresholds = np.zeros([y_size, x_size])

        for x_idx in range(x_size):
            for y_idx in range(y_size):
                thresholds[y_idx, x_idx] = self.compute_local_conf(
                    threshold=threshold,
                    z_ss=(0, z_size),
                    y_ss=self._y_start_stop[y_idx],
                    x_ss=self._x_start_stop[x_idx]
                )
        return thresholds

    def compute_local_conf(self, threshold: int, z_ss: tuple, y_ss: tuple, x_ss: tuple):
        return np.percentile([self._bg_images[z, y, x] for (z, y, x) in zip(
            self._rng.integers(z_ss[0], z_ss[1], size=self._samples),
            self._rng.integers(y_ss[0], y_ss[1], size=self._samples),
            self._rng.integers(x_ss[0], x_ss[1], size=self._samples)
        )], q=threshold)

    def compute_grid(self):
        shape = self._bg_images.shape
        x_len = shape[-1]
        y_len = shape[-2]
        x_slen = int(floor(x_len / self._divisions))
        y_slen = int(floor(y_len / self._divisions))
        x_start_stop = [(x, min(x + x_slen, x_len)) for x in
                        np.arange(0, x_len, x_slen)]
        y_start_stop = [(y, min(y + y_slen, y_len)) for y in
                        np.arange(0, y_len, y_slen)]
        return x_start_stop, y_start_stop

    def mean(self):
        return np.mean(self._bg_images)

    def variance(self):
        return np.var(self._bg_images)

    @staticmethod
    def _consistent_shape(nparr: np.ndarray) -> bool:
        return 1 == len(set([nparr[idx].shape for idx in range(nparr.shape[0])]))
