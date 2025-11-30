import numpy as np
from scipy.optimize import curve_fit

from blur_func import gaussian_blur2d
from data_analysis import get_m_and_g_constants  # used only to build p0 for the nonlinear fit
from find_V_and_phi_tag_tag import V_fun, phi_tag_tag_fun
from guinier_approximation import estimate_Rg

def initial_processing_expo(
    I1,
    chi,
    *,
    I2=None,
    sigma=None
):
    """
    Args:
        I1: a numpy 2D array of intensity in a SAXS image
        chi: a numpy 2D array of the angle of each q value in the SAXS image

        * and we need one of two extra parameters
        I2: a numpy 2D array of a second intensity mesurment in a SAXS image with diffrent anisotropy parameters
        sigma: a scalar, we use it to convole I1 with a gaussian blur of stdv=sigma and use
        # that as our I2. assumed numpy.float

    Returns:
        R: a numpy 2D array of the values I1/I2
        cos_2_chi: a numpy 2D array of cos(2*chi)
    """
    # we first make sure we got relevant data
    if (I2 is None) == (sigma is None):
        raise ValueError("Provide exactly one of I2 or sigma.")

    # if we get sigma instead of I2, we convole I1 to generate a relevant I2
    if sigma is not None:
        I2 = gaussian_blur2d(I1, float(sigma))

    # we can then make sure shape sizes are equal
    if I1.shape != I2.shape or I1.shape != chi.shape:
        raise ValueError("I1, I2, and chi must have identical shapes.")

    # Mask where the ratio is valid and finite
    # np.errstate suppresess two error messages. divide is divinding by 0, by ignoring we get "inf", a string,
    # as the value, while invalid is invalid operations like 0/0, by ignoring we get "nan", a string as well.
    # we ignore them because we are about to mask them anyway, and we don't need an explicit error message each time
    # it happens.
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = I1 / I2

    # we then create a boolean mask out all the data that isn't a finite number
    finite = np.isfinite(ratio)
    # we prepear an output array of the dimensions of I1, filled with nan in each cell.
    R = np.full_like(I1, np.nan, dtype=float)
    # we calculate plug into R values of I1/I2 where the mask applies.
    R[finite] = ratio[finite]
    # we calculate the cosine of 2*chi
    cos_2_chi = np.cos(2 * chi)
    return R, cos_2_chi


def _ratio_model(z, g0, g1, g2, g3, m1, m2, m3):
    """
    Nonlinear model for the *ratio*:
        R = exp( G(t) + M(t) * x )
    where:
        t = q**2 (flattened 1D)
        x = cos(2*chi) (flattened 1D)
        G(t) = g0 + g1*t + g2*t**2 + g3*t**3
        M(t) = m1*t + m2*t**2 + m3*t**3
    """
    # we define the intial parameters as a 2D list
    t, x = z
    G = g0 + g1*t + g2*t**2 + g3*t**3
    M = m1*t + m2*t**2 + m3*t**3
    return np.exp(G + M * x)


def get_m_and_g_constants_expo(
    I1,
    q,
    chi,
    *,
    I2=None,
    sigma=None,
    max_nfev=None
):

    """
    Args:
        I1: a numpy 2D array of intensity in a SAXS image
        q: a numpy 2D array of the MAGNITUDE of each q value in the SAXS image
        chi: a numpy 2D array of the angle of each q value in the SAXS image

        * and we need one of two extra parameters
        I2: a numpy 2D array of a second intensity mesurment in a SAXS image with diffrent anisotropy parameters
        sigma: a scalar, we use it to convole I1 with a gaussian blur of stdv=sigma and use
        that as our I2. assumed numpy.float

        max_nfev, an int, the number of iterations we allow curve_fit to run at most.

    Returns:
        coeff: the coeffients of a relevant polynomial fit, a list of floats.
    """
    """
    Fit the nonlinear ratio-space model:
        R = exp( G(t) + M(t) * x ),  with t = q**2,  x = cos(2*chi)

    
    Steps:
      1) Build (R, x) using initial_processing_expo (no sign restriction on R).
      2) Build t = q**2.
      3) Use the linear log-space solver from data_analysis.get_m_and_g_constants
         as the initial guess p0 = (g0..g3, m1..m3).
      4) Run scipy.optimize.curve_fit to refine all 7 coefficients jointly.

    Returns:
      (g0, g1, g2, g3, m1, m2, m3)  as floats
    """
    # we raise an error if the arrays aren't of the same sizes.
    if I1.shape != q.shape or I1.shape != chi.shape:
        raise ValueError("I1, q, and chi must have identical shapes.")
    # we calculate relevant parameters for our fitting using intial processing, with the optional I2 or sigma
    # 1) ratio & features (2D)
    R2d, x2d = initial_processing_expo(I1, chi, I2=I2, sigma=sigma)

    # 2) build t (2D), then flatten with finite-mask only
    # Build flattened arrays and mask out invalids: the least sqaure fit needs 1D arrays to work.
    # ravel takes the 2D arrays and put all rows one after the other.
    t2d = q**2
    mask = np.isfinite(R2d) & np.isfinite(x2d) & np.isfinite(t2d)
    if not np.any(mask):
        raise ValueError("No finite samples to perform the nonlinear fit.")

    R = R2d[mask].ravel()
    x = x2d[mask].ravel()
    t = t2d[mask].ravel()

    # 3) initial guess from the linear pipeline
    p0 = get_m_and_g_constants(I1, q, chi, I2=I2, sigma=sigma)

    # 4) Nonlinear least squares on the ratio
    popt, _ = curve_fit(_ratio_model, (t, x), R, p0=p0, max_nfev=max_nfev)

    # return the values
    g0, g1, g2, g3, m1, m2, m3 = [float(v) for v in popt]
    return g0, g1, g2, g3, m1, m2, m3


def get_V_and_phi_tagtag(
    I1,
    q,
    chi,
    q_min,
    q_max,
    *,
    I2=None,
    sigma=None,
    max_nfev=10
):
    """
    Args:
        I1: a numpy 2D array of intensity in a SAXS image
        q: a numpy 2D array of the MAGNITUDE of each q value in the SAXS image
        chi: a numpy 2D array of the angle of each q value in the SAXS image
        q_min: minimal q value of a Guinier approximation
        q_max: maximal q value of a Guinier approximation

        * and we need one of two extra parameters
        I2: a numpy 2D array of a second intensity mesurment in a SAXS image with diffrent anisotropy parameters
        sigma: a scalar, we use it to convole I1 with a gaussian blur of stdv=sigma and use
        that as our I2. assumed numpy.float

        max_nfev, an int, the number of iterations we allow curve_fit to run at most.

    Returns:
        values: a list of three value: Rg, V and phi_tag_tag

    """
    """
    Wrapper for the exponential (ratio-space) path:

      - Fit the nonlinear model to get (g0..g3, m1..m3) using get_m_and_g_constants_expo.
      - Compute:
            c_g = g1 / (g0 * r_g_sq_mod)
            c_m = m2 / (m1 * r_g_sq_mod)
      - Map to (V, phi_tag_tag) via the shared polynomial lookups.

    Returns:
      [(V, phi_tag_tag)]
    """
    # using Guininer aprproximation get r_g_sq_mod - the Rg value with the varience
    # we first flatten our q and I values into 1D.
    """r_g_sq_mod: a numpy float, the value of Rg^2(1+V) which is derived from guinner analysis of the data."""
    flat_q = q.flatten()
    flat_I1 = I1.flatten()
    r_g_sq_mod = estimate_Rg(flat_q, flat_I1, q_min, q_max) ^ 2

    # raise rlevant errors
    if r_g_sq_mod <= 0:
        raise ValueError("r_g_sq_mod must be positive.")

    # Fit coefficients (handles I2 generation & masking internally)
    g0, g1, g2, g3, m1, m2, m3 = get_m_and_g_constants_expo(
        I1, q, chi, I2=I2, sigma=sigma, max_nfev=max_nfev
    )
    # Use the right coefficients for c_g and c_m
    c_g = g1 / (g0 * r_g_sq_mod)
    c_m = m2 / (m1 * r_g_sq_mod)
    # derive V and phi_tag_tag using the helper polynomial fit we wrote.
    V = V_fun(c_m, c_g)
    # raise rlevant error
    if V <= 0:
        raise ValueError("V came out negative.")
    phi_tag_tag = phi_tag_tag_fun(c_m, c_g)
    r_g = np.sqrt(r_g_sq_mod / (1 + V))

    return [(float(r_g), float(V), float(phi_tag_tag))]