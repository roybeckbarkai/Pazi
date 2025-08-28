import numpy as np
# in this helper file we define the particular gaussian blur we use. We define it ourselves instead of using
# scipy.signal.convolve2d because the 2D gaussian is separable, allowing us to convolve data PER AXIS, which is
# quicker and less prone for errors then a direct 2D convolution

def gaussian_kernel1d(sigma, truncate=3.0):
    """
    Args:
    sigma: a scalar (as chosen for data analysis). assumed numpy.float
    truncate: a scalar for choosing the length over which we calculate the kernel. assumed numpy.float

    Returns: k: a numpy 1d array of the Gaussian values at a sensible radius of truncate* sigma values
    """

    """
    Build a 1D Gaussian kernel with mean 0 and std=sigma.
    Kernel length = 2*ceil(truncate*sigma)+1, normalized to sum=1.
    """
    # sigma : float -> standard deviation (pixels)
    if sigma <= 0: # sigma must be positive
        raise ValueError("sigma must be positive.")

    # define the radius of values we take. Note that radius is in units of pixels
    # we make sure two things: int - makes sure the radius is an integer as pixel number is discrete. np.ceil is
    # an upward ceiling function [np.ceil(3.1)=4 for example]. This ensures the radius is at least at long as
    # truncate* sigma, and a bit longer if necessary so we never cut the kernel shorter than we intended.
    radius = int(np.ceil(truncate * sigma))

    # we calculate our gaussian only up to the radius, we assume values outside it are so small that they are negligible.
    # therefor - calculating outside the radius is irrelevant for future calculations.

    # we create the array of values over which we calculate the gaussian. x is a 1d numpy array.
    # np.arrange take the start value -radius, the stop value radius+1 (the +1 makes sure we INCLUDE the value of radius.
    # as no step is defined the values go with +1 each time. we define the data as a float so np.exp will be able to
    # take it. for example if radius = 3 than x = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    x = np.arange(-radius, radius + 1, dtype=float)

    # k - a 1d numpy array - the gaussian function without normalizing. calculated over our array of values x.
    k = np.exp(-0.5 * (x / sigma) ** 2)
    # diving k by the sum of all calculated values of k normalizes k such the k.sum() is now equal to 1. instead of
    # using sqrt(2* pi) or stuff like that.
    k /= k.sum()

    # we return k, a numpy 1d array, of values of a normalized gaussian with stdv=sigma, at a reasonable radius
    # to save computing power.
    return k


def convolve1d_reflect(img, kernel, axis):
    """
    Args:
    img: np 2D array representing our image before it is convolved
    axis: an int, 0 or 1, to choose axis for the convolution
    kernel: a np 1D array, representing our kernel

    Returns: an array with the same shape as img - convolved in one direction
    """

    """
    Separable 1D convolution along one axis with 'reflect' padding.
    Returns an array with the same shape as img.
    
    the process is: 1) padding the data to prevent problems as the edges
    2) choose an axis and create many kernel lines in that direction to convolve
    3) convolve each line of data, then reshape it into a 2D array
    """

    # first, we take our image and create padding.

    # we assume a general kernel. taking kernel.size and diving by two (rounding up with a double slash)
    # returns the radius. At the edges we are going to reflect up to radius length. for our general
    # approach we call this number "the pad"
    pad = kernel.size // 2
    # we define an "empty" pad_width. we take the dimensions of the image (img.ndim) and using the tuple
    # define a padding of "0 lines" in each direction.
    pad_width = [(0, 0)] * img.ndim
    # redefining the pad_width, along the axis of convolution we define where the padding take place, extra lines
    # at the beginning and the end with a total number of "pad" lines in each side.
    pad_width[axis] = (pad, pad)

    # using np.pad we take our image, and pad it along the relevant direction with reflected data.
    # reflection is a reasonable way to pad convolution, taking data up to the radius and reflection it, assuming
    # symmetry at the center of the kernel (gaussian) and lack of interest in the edges (little to no use)

    img_padded = np.pad(img, pad_width=pad_width, mode='reflect')

    # separate the data and kernel into relevant lines and organize them for correct indexing
    img_move = np.moveaxis(img_padded, axis, 0)
    rest_shape = img_move.shape[1:]
    rest_size = int(np.prod(rest_shape)) if rest_shape else 1

    img_lines = img_move.reshape(img_move.shape[0], rest_size)
    out_lines = np.empty((img_move.shape[0] - 2 * pad, rest_size), dtype=img.dtype)

    # perform the convolution for each line
    for j in range(rest_size):
        out_lines[:, j] = np.convolve(img_lines[:, j], kernel, mode='valid')

    # reshape the convolved lines into a 2D array
    out_move = out_lines.reshape((img.shape[axis],) + rest_shape)
    return np.moveaxis(out_move, 0, axis)


def gaussian_blur2d(image, sigma, truncate=3.0):
    """
    Args:
    image: np 2D array representing our image before it is convolved
    sigma: a scalar (as chosen for data analysis). assumed numpy.float
    truncate: a scalar for choosing the length over which we calculate the kernel. assumed numpy.float

    Returns:
    out: a numpy 2d array with the same shape as image - convolved in two directions with a gaussian defined
    but sigma and truncate.
    """
    """
    Apply a separable 2D Gaussian blur with std=sigma to 'image' (μ=0).
    Uses reflect padding; returns same shape as input.
    """
    k = gaussian_kernel1d(sigma, truncate=truncate)
    tmp = convolve1d_reflect(image, k, axis=1)  # horizontal
    out = convolve1d_reflect(tmp, k, axis=0)    # vertical
    return out

