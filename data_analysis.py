import numpy as np
from find_V_and_phi_tag_tag import V_fun, phi_tag_tag_fun


def initial_processing(
    I1,
    I2,
    chi
):
    """
    Compute the log-intensity difference and cosine double-angle map.
    """
    # I1 : numpy array -> Intensity map I1
    # I2 : numpy array -> Intensity map I2 (e.g., convolved I1)
    # chi: numpy array -> Angle map in radians

    ln_I1 = np.log(I1)
    ln_I2 = np.log(I2)
    ln_delta_I = ln_I1 - ln_I2
    cos_2_chi = np.cos(2 * chi)
    return ln_delta_I, cos_2_chi


def get_m_and_g_constants(
    I1,
    I2,
    q,
    chi
):
    """
    Global linear least-squares fit over ALL samples for:
      y = G(q) + M(q) * x,
      with t = q**2 and polynomial bases:
        G(q) = g0 + g1*t + g2*t**2 + g3*t**3
        M(q) = m1*t + m2*t**2 + m3*t**3

    Returns:
      g0, g1, m1, m2
    """
    # I1 : numpy array -> Intensity map I1
    # I2 : numpy array -> Intensity map I2
    # q  : numpy array -> Radial q
    # chi: numpy array -> Angle chi

    y, x = initial_processing(I1, I2, chi)  # y = ln(I1) - ln(I2), x = cos(2*chi)
    y = y.ravel()
    x = x.ravel()
    t = (q**2).ravel()

    valid = np.isfinite(y) & np.isfinite(x) & np.isfinite(t)
    y = y[valid]
    x = x[valid]
    t = t[valid]

    # Design matrix: [1, t, t^2, t^3, x*t, x*t^2, x*t^3]
    Phi_g = np.column_stack([np.ones_like(t), t, t**2, t**3])
    Phi_m = np.column_stack([t, t**2, t**3])
    Z = np.column_stack([Phi_g, x[:, None] * Phi_m])

    theta, _, _, _ = np.linalg.lstsq(Z, y, rcond=None)
    g0, g1, g2, g3, m1, m2, m3 = theta
    return float(g0), float(g1), float(m1), float(m2)


def get_V_and_phi_tagtag(
    I1,
    I2,
    q,
    chi,
    r_g_sq_mod
):
    """
    Compute (V, phi_tagtag) using polynomial maps over (c_g, c_m).
    """
    # I1         : numpy array -> Intensity map I1
    # I2         : numpy array -> Intensity map I2
    # q          : numpy array -> Radial q
    # chi        : numpy array -> Angle chi
    # r_g_sq_mod : float       -> Modified radius of gyration squared

    if I1.shape != I2.shape or I1.shape != q.shape or I1.shape != chi.shape:
        raise ValueError("I1, I2, q, and chi must have the same shape.")
    if r_g_sq_mod <= 0:
        raise ValueError("r_g_sq_mod must be positive.")

    # Fit constants once
    g0, g1, m1, m2 = get_m_and_g_constants(I1, I2, q, chi)

    # Dimensionless ratios (new definitions)
    c_g = g1 / (g0 * r_g_sq_mod)
    c_m = m2 / (m1 * r_g_sq_mod)

    # Evaluate polynomial maps (lookup-based)
    V = V_fun(c_m, c_g)
    phi = phi_tag_tag_fun(c_m, c_g)

    # Keep same return shape (list of (V, phi)) for compatibility
    return [(float(V), float(phi))]