import numpy as np

def V_fun(m_coeff, g_coeff):
    """
    Args:
        m_coeff: a scalar number (as derived from data analysis). assumed numpy.float
        g_coeff: a scalar number (as derived from data analysis). assumed numpy.float

    Returns: V: a scalar number (as derived from data analysis). assumed numpy.float
    """

    """
    Compute V from (m_coeff, g_coeff) using embedded polynomial coefficients.
    Supports scalar, list, or NumPy array inputs.
    """
    # m_coeff : scalar/array -> c_m as found in data analysis
    # g_coeff : scalar/array -> c_g as found in data analysis
    m_coeff = np.asarray(m_coeff)
    g_coeff = np.asarray(g_coeff)

    # VCOEFF : numpy 1d array, containing all the polynomial coefficients used to find V

    VCOEFF = np.array([
        -0.0912,  # constant
        -0.3874,  # m
        -0.0167,  # g
        1.6786,   # m^2
        -3.2844,  # m*g
        0.9468,   # g^2
        3.7108,   # m^2*g
        -2.7926   # m*g^2
    ])

    # we plug in the return the coefficient and powers of the polynomial, returning V immediately.

    return (
        VCOEFF[0]
        + VCOEFF[1] * m_coeff
        + VCOEFF[2] * g_coeff
        + VCOEFF[3] * m_coeff**2
        + VCOEFF[4] * (m_coeff * g_coeff)
        + VCOEFF[5] * g_coeff**2
        + VCOEFF[6] * (m_coeff**2 * g_coeff)
        + VCOEFF[7] * (m_coeff * g_coeff**2)
    )


def phi_tag_tag_fun(m_coeff, g_coeff):
    """
     Args:
         m_coeff: a scalar number (as derived from data analysis). assumed numpy.float
         g_coeff: a scalar number (as derived from data analysis). assumed numpy.float

     Returns: phi_tag_tag: a scalar number (as derived from data analysis). assumed numpy.float
     """

    """
    Compute phi_tagtag lookup from (m_coeff, g_coeff) using embedded polynomial coefficients.
    Supports scalar, list, or NumPy array inputs.
    """
    # m_coeff : scalar/array -> c_m as found in data analysis
    # g_coeff : scalar/array -> c_g as found in data analysis
    m_coeff = np.asarray(m_coeff)
    g_coeff = np.asarray(g_coeff)

    # NCOEFF : numpy 1d array, containing all the polynomial coefficients used to find phi_tag_tag

    NCOEFF = np.array([
        0.0421,   # constant
        -0.3678,  # m
        0.2015,   # g
        0.3998,   # m^2
        -0.6854,  # m*g
        0.1611,   # g^2
        0.2701,   # m^2*g
        -0.2889   # m*g^2
    ])

    # we plug in the return the coefficient and powers of the polynomial, returning phi_tag_tag immediately.

    return (
        NCOEFF[0]
        + NCOEFF[1] * m_coeff
        + NCOEFF[2] * g_coeff
        + NCOEFF[3] * m_coeff**2
        + NCOEFF[4] * (m_coeff * g_coeff)
        + NCOEFF[5] * g_coeff**2
        + NCOEFF[6] * (m_coeff**2 * g_coeff)
        + NCOEFF[7] * (m_coeff * g_coeff**2)
    )