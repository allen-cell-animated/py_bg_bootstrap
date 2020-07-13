import numpy as np
from scipy.stats import shapiro, anderson


class Analyzer(object):
    def __init__(self, bg_imgs: np.ndarray):
        """
        This class is a starting point for basic image statistics. In it's current form
        it simply using the distribution of the data itself to enable the user to
        evaluate where in the distribution you are. I have also included some simple
        statistical tests for normality

        Parameters
        ----------
        bg_imgs an ndarray of data
        """
        self._bg_images = bg_imgs
        self._rng = np.random.default_rng()

    def compute_confidence(self, threshold: float):
        """
        Query the data provided to determine the value that corresponds with the
        percentage threshold from the data.

        Parameters
        ----------
        threshold the percentage value desired. threshold=95.0 would be all but the
        top 5% of the data.

        Returns
        -------
        float value for the requested threshold
        """
        return np.percentile(self._bg_images, q=threshold)

    def mean(self):
        """
        Returns
        -------
        float that is the mean of the data
        """
        return np.mean(self._bg_images)

    def variance(self):
        """
        Returns
        -------
        float that is the variance of the data
        """
        return np.var(self._bg_images)

    def shapiro(self, alpha: float = 0.05, please_print: bool = False):
        """
        Perform the Shapiro-Wilks test on the data to evaluate normality

        Parameters
        ----------
        alpha (float) is the desired certainty that the data is normally distributed. Smaller
        alpha means a stronger certainty of normality generally 0.05 is a reasonable
        value.

        Returns
        -------
        bool (True, False) returns True if the test is satisfied (Gaussian Data)
        """
        stat, p = shapiro(self._bg_images)
        print("Statistics=%.3f, p=%.3f" % (stat, p))
        # interpret
        is_gaussian = p > alpha
        if please_print:
            if is_gaussian:
                print("Sample looks Gaussian (fail to reject H0)")
            else:
                print("Sample does not look Gaussian (reject H0)")
        return is_gaussian

    def anderson_darling(self, please_print: bool = False):
        """
        This applies the anderson_darling test to check the data matches a normal
        distribution, https://en.wikipedia.org/wiki/Anderson%E2%80%93Darling_test

        Parameters
        ----------
        please_print (bool) if True print output otherwise run silently

        Returns
        -------
        bool (True, False) Return True if Normally distributed.
        """
        # normality test
        result = anderson(self._bg_images.flatten())
        if please_print:
            print("Statistic: %.3f" % result.statistic)
        normal = True
        p = 0
        for i in range(len(result.critical_values)):
            sl, cv = result.significance_level[i], result.critical_values[i]
            normal &= result.statistic < result.critical_values[i]
            if please_print:
                if result.statistic < result.critical_values[i]:
                    print(
                        "%.3f: %.3f, data looks normal (fail to reject H0)" % (sl, cv)
                    )
                else:
                    print(
                        "%.3f: %.3f, data does not look normal (reject H0)" % (sl, cv)
                    )
        return normal
