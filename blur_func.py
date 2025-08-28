import numpy as np
# ---------- Gaussian blur helpers (separable, no SciPy) ----------

def _gaussian_kernel1d(sigma, truncate=3.0):
    """
    Build a 1D Gaussian kernel with mean 0 and std=sigma.
    Kernel length = 2*ceil(truncate*sigma)+1, normalized to sum=1.
    """
    # sigma : float -> standard deviation (pixels)
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    radius = int(np.ceil(truncate * sigma))
    x = np.arange(-radius, radius + 1, dtype=float)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    return k


def _convolve1d_reflect(arr, kernel, axis):
    """
    Separable 1D convolution along one axis with 'reflect' padding.
    Returns an array with the same shape as arr.
    """
    pad = kernel.size // 2
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (pad, pad)
    arr_padded = np.pad(arr, pad_width=pad_width, mode='reflect')

    arr_move = np.moveaxis(arr_padded, axis, 0)
    rest_shape = arr_move.shape[1:]
    rest_size = int(np.prod(rest_shape)) if rest_shape else 1

    arr_lines = arr_move.reshape(arr_move.shape[0], rest_size)
    out_lines = np.empty((arr_move.shape[0] - 2 * pad, rest_size), dtype=arr.dtype)

    for j in range(rest_size):
        out_lines[:, j] = np.convolve(arr_lines[:, j], kernel, mode='valid')

    out_move = out_lines.reshape((arr.shape[axis],) + rest_shape)
    return np.moveaxis(out_move, 0, axis)


def gaussian_blur2d(image, sigma, truncate=3.0):
    """
    Apply a separable 2D Gaussian blur with std=sigma to 'image' (μ=0).
    Uses reflect padding; returns same shape as input.
    """
    k = _gaussian_kernel1d(sigma, truncate=truncate)
    tmp = _convolve1d_reflect(image, k, axis=1)  # horizontal
    out = _convolve1d_reflect(tmp, k, axis=0)    # vertical
    return out

