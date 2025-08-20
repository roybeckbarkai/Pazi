import numpy as np
from typing import Tuple, List


def initial_processing(
    I1: np.ndarray,
    I2: np.ndarray,
    chi: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the log-intensity difference and cosine double-angle map,
    with intensity clamped to avoid log of zero or infinities.

    Parameters
    ----------
    I1 : (M,N) array
        Intensity map I1.
    I2 : (M,N) array
        Intensity map I2 (e.g., convolved I1).
    chi : (M,N) array
        Angle map in radians.

    Returns
    -------
    ln_delta_I : (M,N) array
        ln(I1) - ln(I2), with regions of no-signal forced to 0.
    cos_2_chi : (M,N) array
        cos(2*chi).
    """
    eps = 1e-10
    I1_safe = np.clip(I1, eps, None)
    I2_safe = np.clip(I2, eps, None)

    ln_I1 = np.log(I1_safe)
    ln_I2 = np.log(I2_safe)
    ln_delta_I = ln_I1 - ln_I2

    both_floor = (I1_safe == eps) & (I2_safe == eps)
    ln_delta_I[both_floor] = 0.0

    cos_2_chi = np.cos(2 * chi)
    return ln_delta_I, cos_2_chi


def get_M_and_G(
    I1: np.ndarray,
    I2: np.ndarray,
    q: np.ndarray,
    chi: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute M(q) and G(q) for each unique q using a physics-inspired approach:
      - G(q) = mean of ln_delta_I at that q
      - M(q) = slope of residuals vs cos(2chi) through the origin

    Parameters
    ----------
    I1, I2, q, chi : (M,N) arrays
        Intensity maps, radial q, and angle chi.

    Returns
    -------
    M_values : (K,) array
        Slope coefficients M for each unique q.
    G_values : (K,) array
        Intercept values G for each unique q.
    q_values : (K,) array
        Sorted unique q values.
    """
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
    I1: np.ndarray,
    I2: np.ndarray,
    q: np.ndarray,
    chi: np.ndarray
) -> Tuple[float, float, float, float]:
    """
    Fit polynomial constants for G(q) and M(q):
      G(q) = g0 + g1*q^2 + g2*q^4 + g3*q^6
      M(q) = m1*q^2 + m2*q^4 + m3*q^6

    Returns
    -------
    g0, g1, m1, m2 : float
        Polynomial coefficients of interest.
    """
    M_values, G_values, q_values = get_M_and_G(I1, I2, q, chi)
    t = q_values**2
    coeffs_g = np.polyfit(t, G_values, 3)
    g3, g2, g1, g0 = coeffs_g
    A = np.vstack([t, t**2, t**3]).T
    m1, m2, m3 = np.linalg.lstsq(A, M_values, rcond=None)[0]
    return g0, g1, m1, m2


def get_V_and_phi_tagtag(
    I1: np.ndarray,
    I2: np.ndarray,
    q: np.ndarray,
    chi: np.ndarray,
    r_g: float
) -> List[Tuple[float, float]]:
    """
    Analytically solve for possible (V, phi_tagtag) roots based on dimensionless ratios.

    Parameters
    ----------
    I1, I2, q, chi : (M,N) arrays
        Intensity and geometry data.
    r_g : float
        Radius of gyration.

    Returns
    -------
    solutions : list of (V, phi_tagtag) tuples
        All real, positive solutions satisfying the derived equations.
    """
    # Validate inputs
    if I1.shape != I2.shape or I1.shape != q.shape or I1.shape != chi.shape:
        raise ValueError("I1, I2, q, and chi must have the same shape.")
    if r_g <= 0:
        raise ValueError("r_g must be positive.")

    # Compute coefficients
    g0, g1, m1, m2 = get_m_and_g_constants(I1, I2, q, chi)
    c_m = (m2 / m1) * (r_g ** 2)
    c_g = (g1 / g0) * (r_g ** 2)

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
