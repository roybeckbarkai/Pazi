import numpy as np


def gaussian(x, mean=0, std=1):
    """
    Normalized Gaussian probability density function.

    Parameters
    ----------
    x : np.ndarray
        Input values.
    mean : float
        Center of the Gaussian.
    std : float
        Standard deviation.

    Returns
    -------
    pdf : np.ndarray
        Probability density values, sum normalized to 1.
    """
    pdf = np.exp(-(x - mean) ** 2 / (2 * std ** 2))
    pdf /= pdf.sum()  # normalize so that sum(pdf) == 1
    return pdf
