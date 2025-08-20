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
    Compute global M and G using all samples (no q-binning):
      - G = mean of ln_delta_I over all samples
      - M = slope (through origin) of residuals vs cos(2chi) over all samples
    """
    # I1 : numpy array -> Intensity map I1
    # I2 : numpy array -> Intensity map I2
    # q  : numpy array -> Radial q (unused for binning here; kept for interface)
    # chi: numpy array -> Angle chi

    ln_delta_I, cos_2_chi = initial_processing(I1, I2, chi)

    # Flatten and drop NaNs
    y = ln_delta_I.ravel()
    x = cos_2_chi.ravel()
    valid = ~np.isnan(y) & ~np.isnan(x)
    y = y[valid]
    x = x[valid]

    # G: mean of ln(I1) - ln(I2)
    G = np.nanmean(y)

    # M: slope through origin of (y - G) vs x
    y_res = y - G
    den = np.sum(x**2)
    M = np.sum(x * y_res) / den if den != 0 else np.nan

    return M, G


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