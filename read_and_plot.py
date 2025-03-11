import save_and_load as csv_man
import numpy as np
import matplotlib.pyplot as plt
def G_function(x, rg_fit, f2_fit, var_fit, A_fit):
    """
    Compute the G(q) function for fitting log intensity ratios.

    This function calculates an expansion in q with four terms (g0 to g3)
    and applies an overall scaling factor. The expansion parameters depend
    on the fitted radius of gyration (rg_fit), f2 parameter (f2_fit), variance (var_fit),
    and a scaling factor (A_fit).

    Parameters
    ----------
    x : array_like
        The q values (scattering vector).
    rg_fit : float
        The fitted radius of gyration.
    f2_fit : float
        The fitted f2 parameter (related to the form factor).
    var_fit : float
        The fitted variance parameter.
    A_fit : float
        The scaling factor.

    Returns
    -------
    array_like
        The computed G(q) values.
    """
    # Precompute powers of x and rg for clarity and efficiency
    x2 = x**2
    x4 = x**4
    x6 = x**6
    rg2 = rg_fit**2
    rg4 = rg_fit**4
    rg6 = rg_fit**6

    # Compute expansion terms
    g0 = -(1 + var_fit)
    g1 = (2/3 - 18 * f2_fit + (16/3 - 108 * f2_fit) * var_fit) * x2 * rg2
    g2 = (8 * f2_fit + (176 * f2_fit - 16/9) * var_fit) * x4 * rg4
    g3 = (24 * f2_fit**2 + (960 * f2_fit**2 - (128/3) * f2_fit) * var_fit) * x6 * rg6

    # Combine terms and apply the overall scaling factor
   # return (2/3) * rg2 * A_fit * (g0 + g1 + g2 + g3)
    return A_fit * (g0 + g1 + g2 + g3)

q1_binned,logdI = csv_man.read_q_I_from_csv ("q_log(unfilt)-log(filt).csv")
plt.plot(q1_binned**2, logdI, 'bo', label='Data')

rg_initial=2
f2_initial=0
var_initial=0.001

logdI_zero = logdI[0]
A_initial = logdI_zero / G_function(0, rg_initial, f2_initial, var_initial, 1)
G_initial = G_function(q1_binned, rg_initial, f2_initial, var_initial, A_initial)

plt.plot(q1_binned**2, G_initial, 'k--', label='Initial parameters')
plt.xlabel('q^2')
plt.ylabel('ln(I1/I2)')