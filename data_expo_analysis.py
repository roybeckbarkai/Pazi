import numpy as np
from find_V_and_phi_tag_tag import V_fun, phi_tag_tag_fun
from scipy.optimize import curve_fit
from blur_func import gaussian_blur2d

# ---------------------- Nonlinear (exp) Pipeline ----------------------

def _prepare_ratio_and_features(
    I1,
    q,
    chi,
    *,
    I2=None,
    sigma=None
):
    """
    Prepare valid ratio R=I1/I2 (>0, finite), feature t=q**2, and x=cos(2*chi).
    If I2 is None and sigma is provided, generate I2 by Gaussian-blur of I1.
    """
    # I1  : numpy array -> Intensity map I1
    # q   : numpy array -> Radial q (same shape as I1)
    # chi : numpy array -> Angle map in radians (same shape as I1)
    # I2  : numpy array or None -> Optional intensity map I2
    # sigma : float or None -> std to generate I2 via Gaussian blur

    if I1.shape != q.shape or I1.shape != chi.shape:
        raise ValueError("I1, q, and chi must have identical shapes.")
    if (I2 is None) == (sigma is None):
        raise ValueError("Provide exactly one of I2 or sigma for ratio building.")

    if sigma is not None:
        I2 = gaussian_blur2d(I1, float(sigma))
    if I1.shape != I2.shape:
        raise ValueError("I1 and I2 must have identical shapes.")

    with np.errstate(divide='ignore', invalid='ignore'):
        R = I1 / I2
    valid = np.isfinite(R) & (R > 0)

    # Flatten and mask
    R = R.ravel()
    x = np.cos(2 * chi).ravel()
    t = (q**2).ravel()

    mask = valid.ravel() & np.isfinite(x) & np.isfinite(t)
    R = R[mask]; x = x[mask]; t = t[mask]

    if R.size == 0:
        raise ValueError("No valid samples (I1/I2 > 0) to perform the fit.")

    return R, t, x


def _linear_log_initial_guess(
    I1,
    q,
    chi,
    *,
    I2=None,
    sigma=None
):
    """
    Get an initial guess for (g0,g1,g2,g3,m1,m2,m3) via the linear log-fit:
        ln(R) ≈ G(t) + M(t)*x,
    with t = q**2, x = cos(2*chi). This solves a single linear least-squares
    for the 7 coefficients.
    """
    # Build ratio and mask
    if (I2 is None) == (sigma is None):
        raise ValueError("Provide exactly one of I2 or sigma for the initial-guess pipeline.")

    # Construct I2 if needed and compute ratio
    if sigma is not None:
        I2 = gaussian_blur2d(I1, float(sigma))
    if I1.shape != q.shape or I1.shape != chi.shape or I1.shape != I2.shape:
        raise ValueError("I1, I2, q, and chi must have identical shapes.")

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = I1 / I2
    valid = np.isfinite(ratio) & (ratio > 0)

    y = np.full_like(I1, np.nan, dtype=float)
    y[valid] = np.log(ratio[valid])
    x = np.cos(2 * chi)

    # Flatten & mask
    y = y.ravel()
    x = x.ravel()
    t = (q**2).ravel()

    mask = np.isfinite(y) & np.isfinite(x) & np.isfinite(t)
    y = y[mask]; x = x[mask]; t = t[mask]

    if y.size == 0:
        raise ValueError("No valid samples (I1/I2 > 0) for initial guess.")

    # Design matrix columns: [1, t, t^2, t^3, x*t, x*t^2, x*t^3]
    Phi_g = np.column_stack([np.ones_like(t), t, t**2, t**3])
    Phi_m = np.column_stack([t, t**2, t**3])
    Z = np.column_stack([Phi_g, x[:, None] * Phi_m])

    theta, _, _, _ = np.linalg.lstsq(Z, y, rcond=None)
    # Return as float tuple for curve_fit p0
    return tuple(float(v) for v in theta)  # (g0,g1,g2,g3,m1,m2,m3)


def _ratio_model((t, x), g0, g1, g2, g3, m1, m2, m3):
    """
    Nonlinear model for ratio:
        R = exp( G(t) + M(t) * x )
    where:
        G(t) = g0 + g1*t + g2*t^2 + g3*t^3
        M(t) = m1*t + m2*t^2 + m3*t^3
    """
    # t, x : 1D arrays (aligned)
    G = g0 + g1*t + g2*t**2 + g3*t**3
    M = m1*t + m2*t**2 + m3*t**3
    return np.exp(G + M * x)


def get_m_and_g_constants_exp(
    I1,
    q,
    chi,
    *,
    I2=None,
    sigma=None,
    bounds=None,
    max_nfev=None
):
    """
    Nonlinear fit on the ratio R = I1/I2:
        R ≈ exp( G(t) + M(t) * x ),  with t = q**2, x = cos(2*chi)

    Steps:
      1) Build (R,t,x) using only valid (R>0) samples.
      2) Compute an initial guess (g0..g3, m1..m3) from the linear log-fit.
      3) Run scipy.optimize.curve_fit to refine all 7 coefficients.

    Options:
      - bounds: pass (lower_bounds, upper_bounds) if desired (each length-7).
      - max_nfev: limit iterations for curve_fit.

    Returns ALL 7 coefficients:
      (g0, g1, g2, g3, m1, m2, m3)
    """
    # I1 : numpy array -> Intensity map I1
    # q  : numpy array -> Radial q (same shape as I1)
    # chi: numpy array -> Angle chi (same shape as I1)
    # I2 : numpy array or None
    # sigma : float or None
    # bounds : None or tuple (lb, ub) each shape (7,)
    # max_nfev : None or int

    # 1) Prepare data
    R, t, x = _prepare_ratio_and_features(I1, q, chi, I2=I2, sigma=sigma)

    # 2) Initial guess via linear log-fit (robust & fast)
    p0 = _linear_log_initial_guess(I1, q, chi, I2=I2, sigma=sigma)

    # 3) Nonlinear LS on ratio
    # curve_fit expects xdata as a single object; pass a tuple (t,x)
    if bounds is None:
        popt, _ = curve_fit(_ratio_model, (t, x), R, p0=p0, max_nfev=max_nfev)
    else:
        popt, _ = curve_fit(_ratio_model, (t, x), R, p0=p0, bounds=bounds, max_nfev=max_nfev)

    g0, g1, g2, g3, m1, m2, m3 = [float(v) for v in popt]
    return g0, g1, g2, g3, m1, m2, m3


def get_V_and_phi_tagtag_exp(
    I1,
    q,
    chi,
    *,
    r_g_sq_mod,
    I2=None,
    sigma=None,
    bounds=None,
    max_nfev=None
):
    """
    Convenience wrapper:
      - Fit the nonlinear (exp) model to get all coefficients.
      - Build c_g = g1 / (g0 * r_g_sq_mod),  c_m = m2 / (m1 * r_g_sq_mod).
      - Map to (V, phi_tagtag) via lookup polynomials.

    Returns:
      [(V, phi_tagtag)]
    """
    # I1         : numpy array -> Intensity map I1
    # q          : numpy array -> Radial q
    # chi        : numpy array -> Angle chi
    # r_g_sq_mod : float       -> Modified radius of gyration squared (positive)
    # I2, sigma  : optional (mutually exclusive) inputs to define the ratio
    # bounds     : optional curve_fit bounds for the 7-parameter vector
    # max_nfev   : optional iteration cap for curve_fit

    if r_g_sq_mod <= 0:
        raise ValueError("r_g_sq_mod must be positive.")

    g0, g1, g2, g3, m1, m2, m3 = get_m_and_g_constants_exp(
        I1, q, chi, I2=I2, sigma=sigma, bounds=bounds, max_nfev=max_nfev
    )

    c_g = g1 / (g0 * r_g_sq_mod)
    c_m = m2 / (m1 * r_g_sq_mod)

    V = V_fun(c_m, c_g)
    phi = phi_tag_tag_fun(c_m, c_g)
    return [(float(V), float(phi))]