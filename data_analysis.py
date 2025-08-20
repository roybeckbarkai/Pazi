import numpy as np

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

    # Build regression targets/features from ALL samples
    y, x = initial_processing(I1, I2, chi)  # y = ln(I1) - ln(I2), x = cos(2*chi)
    y = y.ravel()
    x = x.ravel()
    t = (q**2).ravel()

    valid = np.isfinite(y) & np.isfinite(x) & np.isfinite(t)
    y = y[valid]
    x = x[valid]
    t = t[valid]

    # Design matrix columns: [1, t, t^2, t^3, x*t, x*t^2, x*t^3]
    Phi_g = np.column_stack([np.ones_like(t), t, t**2, t**3])
    Phi_m = np.column_stack([t, t**2, t**3])
    Z = np.column_stack([Phi_g, x[:, None] * Phi_m])

    theta, _, _, _ = np.linalg.lstsq(Z, y, rcond=None)
    g0, g1, g2, g3, m1, m2, m3 = theta

    # We only need a subset for downstream use
    return float(g0), float(g1), float(m1), float(m2)


def get_V_and_phi_tagtag(
    I1,
    I2,
    q,
    chi,
    r_g_mod
):
    """
    Analytically solve for possible (V, phi_tagtag) roots based on dimensionless ratios.
    """
    # I1      : numpy array -> Intensity map I1
    # I2      : numpy array -> Intensity map I2
    # q       : numpy array -> Radial q
    # chi     : numpy array -> Angle chi
    # r_g_mod : float       -> Radius of gyration (modified)

    # Validate inputs
    if I1.shape != I2.shape or I1.shape != q.shape or I1.shape != chi.shape:
        raise ValueError("I1, I2, q, and chi must have the same shape.")
    if r_g_mod <= 0:
        raise ValueError("r_g_mod must be positive.")

    # Get needed constants directly from the global fit
    g0, g1, m1, m2 = get_m_and_g_constants(I1, I2, q, chi)

    # Dimensionless ratios
    c_m = (m2 / m1) * (r_g_mod ** 2)
    c_g = (g1 / g0) * (r_g_mod ** 2)

    # Quadratic for phi_tagtag: A*phi^2 + B*phi + C = 0
    A = 1404
    B = 102 - 36*c_m + 324*c_g + 30*c_m*c_g
    C = 8/3 - 4*c_m + 8*c_g + 15*c_m*c_g

    coeffs = [A, B, C]
    phis = np.roots(coeffs)

    solutions = []
    for phi in phis:
        phi_val = np.real(phi)
        if np.isclose(phi.imag, 0.0):
            numerator = -1 - 18*phi_val - 3*c_g
            denominator = 10 + 108*phi_val + 3*c_g
            if not np.isclose(denominator, 0.0):
                V = numerator / denominator
                if np.isreal(V) and V > 0:
                    solutions.append((float(V), float(phi_val)))

    return solutions