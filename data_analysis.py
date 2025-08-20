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


def get_M_and_G(
    I1,
    I2,
    q,
    chi
):
    """
    Compute M(q) and G(q) for each unique q using a physics-inspired approach:
      - G(q) = mean of ln_delta_I at that q
      - M(q) = slope of residuals vs cos(2chi) through the origin
    """
    # I1 : numpy array -> Intensity map I1
    # I2 : numpy array -> Intensity map I2
    # q  : numpy array -> Radial q
    # chi: numpy array -> Angle chi

    ln_delta_I, cos_2_chi = initial_processing(I1, I2, chi)
    q_values = np.unique(q)
    K = q_values.size
    M_values = np.zeros(K)
    G_values = np.zeros(K)

    for idx, q_i in enumerate(q_values):
        mask = (q == q_i)
        y = ln_delta_I[mask]
        x = cos_2_chi[mask]
        valid = ~np.isnan(y)
        y = y[valid]
        x = x[valid]

        G_i = np.nanmean(y)
        num = np.sum(x * (y - G_i))
        den = np.sum(x**2)
        M_i = num / den if den != 0 else np.nan

        G_values[idx] = G_i
        M_values[idx] = M_i

    return M_values, G_values, q_values


def get_m_and_g_constants(
    I1,
    I2,
    q,
    chi
):
    """
    Fit polynomial constants for G(q) and M(q):
      G(q) = g0 + g1*q^2 + g2*q^4 + g3*q^6
      M(q) = m1*q^2 + m2*q^4 + m3*q^6
    """
    # I1 : numpy array -> Intensity map I1
    # I2 : numpy array -> Intensity map I2
    # q  : numpy array -> Radial q
    # chi: numpy array -> Angle chi

    M_values, G_values, q_values = get_M_and_G(I1, I2, q, chi)
    t = q_values**2
    coeffs_g = np.polyfit(t, G_values, 3)
    g3, g2, g1, g0 = coeffs_g
    A = np.vstack([t, t**2, t**3]).T
    m1, m2, m3 = np.linalg.lstsq(A, M_values, rcond=None)[0]
    return g0, g1, m1, m2


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

    # Compute coefficients
    g0, g1, m1, m2 = get_m_and_g_constants(I1, I2, q, chi)
    c_m = (m2 / m1) * (r_g_mod ** 2)
    c_g = (g1 / g0) * (r_g_mod ** 2)

    # Quadratic for phi_tagtag: A*phi^2 + B*phi + C = 0
    A = 1404
    B = 102 - 36*c_m + 324*c_g + 30*c_m*c_g
    C = 8/3 - 4*c_m + 8*c_g + 15*c_m*c_g

    # Use numpy.roots for numerical stability
    coeffs = [A, B, C]
    phis = np.roots(coeffs)

    solutions = []
    for phi in phis:
        phi_val = np.real(phi)
        # Select real solutions within tolerance
        if np.isclose(phi.imag, 0.0):
            # Compute V from phi and c_g
            numerator = -1 - 18*phi_val - 3*c_g
            denominator = 10 + 108*phi_val + 3*c_g
            if not np.isclose(denominator, 0.0):
                V = numerator / denominator
                # Keep only real, positive V
                if np.isreal(V) and V > 0:
                    solutions.append((float(V), float(phi_val)))

    return solutions