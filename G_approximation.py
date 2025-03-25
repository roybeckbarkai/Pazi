import numpy as np
#import lmfit
from typing import Optional
import random
import guinier_approximation

import save_and_load as csv_man

from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# Defines G(q) fitting function
#List of f2 values based on the form factor

def G_function_new(x, Rg, C2, V, A):
    """
    Compute G(q) = ln(I1(q)/I2(q)) = A*(g0 + g1*Q^2 + g2*Q^4 + g3*Q^6)
    where the coefficients are defined as follows:
      x=q
      Q=(q*Rg)^2  
      g0 = -3* (1 + V)
      g1  = (1+18*C2+ 6*(9*C2 +1)*V)
      g2 = -(6*C2+ V*(72*C2+4/3))
      g3 = (9/2)*C2+(315*C2^2 + 2C2)*V
      
    Parameters
    ----------
    x=q : float or numpy.ndarray
        The radial scattering variable.
    A : float
        The prefactor, representing (sigma_x1^2+sigma_y1^2) - (sigma_x2^2+sigma_y2^2).
    Rg : float
        The mean radius of gyration (R_0).
    V : float
        The variance of the radius of gyration distribution.
    C2 : float
        a constant which is the second derivating with respect to Q of the unsmeared form factor
        
    Returns
    -------
    G : float or numpy.ndarray
        The computed value of ln(I1(q)/I2(q)).
    """
    
    A = float(A)
    Rg = float(Rg)
    V = float(V) / Rg**2  
    C2 = float(C2)
       
    if Rg <= 0:
        raise ValueError("Rg (mean radius of gyration) must be positive.")
    if V < 0:
        raise ValueError("Variance V must be nonnegative.")
    
    # Compute Q = q * Rg.
    Q = (np.array(x, dtype=float) * Rg) ** 2
    if np.any(Q < 0):
        raise ValueError("All q values must be nonnegative.")
    
    # Compute the coefficients.
    g0 = -3 * (1 + V)
    g1 = 1 + 18 * C2 + 6 * (9 * C2 + 1) * V
    g2 = -(6 * C2 + V * (72 * C2 + 4/3))
    g3 = (9/2) * C2**2 + (315 * C2**2 + 2 * C2) * V
    
    # Calculate G(q) = A*(g0 + g1*Q + g2*Q^2 + g3*Q^3).
    G = A * (g0 + g1 * Q + g2 * Q**2 + g3 * Q**3)
    return G
    
   
def G_function_2d_old(x, Rg, C2, V, A):
    """
    Compute G(q) = ln(I1(q)/I2(q)) = A*(g0 + g1*q^2 + g2*q^4 + g3*q^6)
    where the coefficients are defined as follows:
    
      g0 = -2/3*(Rg^2 + V)
      B  = C2*Rg^4 + (6*C2 - 2/9)*Rg^2*V
      g1 = 2/9*(Rg^2 + V)**2 - 8*B
      g2 = 8/3 * B*(Rg^2 + V)
      g3 = 8 * B**2
    
    Parameters
    ----------
    x=q : float or numpy.ndarray
        The radial scattering variable.
    A : float
        The prefactor, representing (sigma_x1^2+sigma_y1^2) - (sigma_x2^2+sigma_y2^2).
    Rg : float
        The mean radius of gyration (R_0).
    V : float
        The variance of the radius of gyration distribution.
    C2 : float
        The constant c2 in the unsmeared intensity.
        
    Returns
    -------
    G : float or numpy.ndarray
        The computed value of ln(I1(q)/I2(q)).
    """
    q = np.array(x, dtype=float)
    if np.any(q < 0):
        raise ValueError("All q values must be nonnegative.")
    A = float(A)
    Rg = float(Rg)
    V = float(V)
    C2 = float(C2)
       
    if Rg <= 0:
        raise ValueError("Rg (mean radius of gyration) must be positive.")
    if V < 0:
        raise ValueError("Variance V must be nonnegative.")
    
    # Coefficient g0:
    g0 = - (2.0 / 3.0) * (Rg**2 + V)
    
    # Define B:
    B = C2 * (Rg**4) + (6.0 * C2 - 2.0/9.0) * (Rg**2) * V
    
    # Coefficients g1, g2, g3:
    g1 = (2.0 / 9.0) * (Rg**2 + V)**2 - 8.0 * B
    g2 = (8.0 / 3.0) * B * (Rg**2 + V)
    g3 = 8.0 * (B**2)
    
    # Evaluate and return G(q)
    return A * (g0 + g1 * q**2 + g2 * q**4 + g3 * q**6)
    
   
# G(q) fitting function
# x is an argument (q)
# rg_fit -- Rg
# f2_fit -- f2 parameter
# var_fit -- variance
# A_fit -- scaling factor
def G_function_1D(x, rg_fit, f2_fit, var_fit, A_fit):
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

'''
Fits the 1D data with G function
If form_factor_name is given the algorithm will use known f2
If form_factor_name is NOT given the algorithm will fit f2 as free parameter

Input parameters:

Arbitrary parameters:
- q1, I1, q2, I2 -- two sets of Intensities vs q

Optional parameters:

* Rg
-- rg_initial -- Default: 
-- rg_free -- Default: True
-- rg_min, rg_max -- Default: 
-- perform_guinier_estimation -- Default: False
        if True, tries to use Guinier approximation to find initial Rg

* Variance 
-- var_initial: Default randomized in (0,1) range
-- var_free: Default True
-- var_min, var_max: Default in range [0,1]

* f2
-- f2_initial can be either defined by form_factor_name, or manually, or not defined 
-- form_factor_name -- used to define f2 parameter value:  
        guinier_ff: Guinier form factorm, f2 == 0
        sphere_ff: for spherical particles, f2 == 1/126
        gaussian_ff: for gaussian chain, f2 == -1/36
-- f2_free -- Default: True, if True, treated as a free parameter. If False -- fixed.
-- f2_min, f2_max -- Default in fange [-1,1] 
 
 * A (scale factor of G function)
-- A_initial -- avoid 0 values. If 0, will be changed to 1
-- A_free -- Default: True
-- A_min, A_max

-- q_min, q_max -- Default: in full q range of the input data

-- fitting_method -- selection of fitting methods from .lmfit library. Default: 'leastsq' 

-- save_to_log -- optional saving  of the fit results to log file, Default: False
-- log_file_name -- Default: 'auto_log_file'
'''


# Dictionary for f2 values based on form factor name
def adjust_bounds(lower_bounds, upper_bounds, epsilon=1e-8):
    """
    Adjusts the lower and upper bounds to ensure that each lower bound is strictly less than its upper bound.
    
    If a lower bound is greater than or equal to the upper bound, epsilon is subtracted from the lower bound
    and added to the upper bound.
    
    Parameters
    ----------
    lower_bounds : list or tuple of floats
        The lower bounds for the parameters.
    upper_bounds : list or tuple of floats
        The upper bounds for the parameters.
    epsilon : float, optional
        The small value to adjust the bounds by (default is 1e-8).
    
    Returns
    -------
    new_lower : list of floats
        The adjusted lower bounds.
    new_upper : list of floats
        The adjusted upper bounds.
    """
    new_lower = []
    new_upper = []
    for lb, ub in zip(lower_bounds, upper_bounds):
        if lb >= ub:
            lb_adjusted = lb - epsilon
            ub_adjusted = ub + epsilon
            new_lower.append(lb_adjusted)
            new_upper.append(ub_adjusted)
        else:
            new_lower.append(lb)
            new_upper.append(ub)
    return new_lower, new_upper

def calculate_errors(fit_results):
    """Calculate the standard errors from the covariance matrix."""
    cov = fit_results.get("covariance", None)
    params = fit_results.get("optimal_parameters", {})

    if cov is not None:
        errors = np.sqrt(np.diag(cov))
    else:
        errors = [None] * len(params)

    return errors

def print_fitted_results(fit_results):
    """Print the fitted parameters along with their errors."""
    params = fit_results.get("optimal_parameters", {})
    errors = calculate_errors(fit_results)
    
    param_order = ["rg_fit", "f2_fit", "var_fit", "A_fit"]

    print("Fitted Parameters (value ± error):")
    for i, param in enumerate(param_order):
        value = params.get(param, None)
        error = errors[i] if errors[i] is not None else "N/A"
        print(f"  {param}: {value:.6g} ± {error:.6g}" if isinstance(error, float) else f"  {param}: {value} ± {error}")



import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.optimize import curve_fit
from typing import Optional

def G_fit(q1, I1, q2, I2,
          form_factor_name: Optional[str] = 'NaN',
          f2_initial: Optional[float] = 0, f2_min: Optional[float] = -1, f2_max: Optional[float] = 1, f2_free: Optional[bool] = True,
          q_min: Optional[float] = None, q_max: Optional[float] = None,
          rg_initial: Optional[float] = 0, rg_min: Optional[float] = 0, rg_max: Optional[float] = 1e5, rg_free: Optional[bool] = True,
          var_initial: Optional[float] = 0, var_min: Optional[float] = 0, var_max: Optional[float] = 1, var_free: Optional[bool] = True,
          A_initial: Optional[float] = 0, A_min: Optional[float] = -1e38, A_max: Optional[float] = 1e38, A_free: Optional[bool] = True,
          perform_guinier_estimation: Optional[bool] = False,
          plot_fitting_curve: Optional[bool] = False,  # Currently unused in this version
          maxfev: Optional[int] = 1e7,
          auto_init_A: Optional[bool] = True,
          auto_set_parameters: Optional[bool] = True,
          auto_rg_bound_percent: Optional[float] = 0.1
          ):
    """
    Fits the log intensity ratio data using the G_function model and SciPy's curve_fit.

    The function processes two data sets (q1, I1) and (q2, I2) to compute:

        logdI = ln(I1 / I2)

    within a chosen q-range, then fits it with G_function (defined elsewhere).
    The resulting fit parameters and masked data are returned.

    Parameters
    ----------
    q1, I1, q2, I2 : array_like
        Two sets of intensities vs. q (1D arrays).
    form_factor_name : str, optional
        Name of the form factor to determine f2. If 'NaN' or unrecognized,
        f2 is treated as a free parameter.
    f2_initial, f2_min, f2_max : float, optional
        Initial guess and bounds for the f2 parameter.
    f2_free : bool, optional
        If True, f2 is allowed to vary; if False, f2 is fixed at f2_initial.
    q_min, q_max : float, optional
        q-range to use for fitting. If None, the range is inferred from q2.
    rg_initial, rg_min, rg_max : float, optional
        Initial guess and bounds for the radius of gyration rg.
    rg_free : bool, optional
        If True, rg is allowed to vary; if False, rg is fixed.
    var_initial, var_min, var_max : float, optional
        Initial guess and bounds for the variance parameter.
    var_free : bool, optional
        If True, variance is allowed to vary; if False, variance is fixed.
    A_initial, A_min, A_max : float, optional
        Initial guess and bounds for the scaling factor A.
    A_free : bool, optional
        If True, A is allowed to vary; if False, A is fixed.
    perform_guinier_estimation : bool, optional
        If True, uses a Guinier approximation to estimate rg from the data.
    plot_fitting_curve : bool, optional
        Currently unused in this code snippet. If True, you may want to add code
        to visualize the fit. 
    maxfev : int, optional
        Maximum number of function evaluations for curve_fit.
    auto_set_parameters : bool, optional
        If True, automatically sets rg_initial using Guinier, adjusts rg bounds,
        and calculates A_initial, A_min, and A_max accordingly.
    auto_rg_bound_percent : float, optional
        If auto_set_parameters is True, controls how much rg can vary around
        its Guinier estimate.

    Returns
    -------
    fit_results : dict
        A dictionary containing the fitted parameters, their errors, and the covariance matrix.
    q1_masked : ndarray
        The q-values used in the fit (after applying the q-range mask).
    logdI : ndarray
        The log intensity ratio ln(I1/I2) used in the fit.
    G_final : ndarray
        The model values G_function(q1_masked, *popt) using the fitted parameters.
    G_initial : ndarray
        The model values G_function(q1_masked, rg_initial, f2_initial, var_initial, A_initial).

    Notes
    -----
    - This function assumes that G_function, adjust_bounds, and 
      guinier_approximation.estimate_Rg are defined elsewhere.
    - For best results, ensure that q1 and q2 share the same q-points or are 
      appropriately interpolated or binned beforehand.
    """
    epsilon = 1e-14

    # Warn if q arrays differ
    if not np.array_equal(q1, q2):
        print('q ranges of the data sets do not match. Consider binning or interpolating the data.')
        return -1

    # Determine default q-range if none provided
    if q_min is None:
        q_min = np.min(q2)
    if q_max is None:
        q_max = np.max(q2)

    # Create a mask for the chosen q-range
    q1 = np.array(q1).flatten()
    I1 = np.array(I1).flatten()
    I2 = np.array(I2).flatten()
    maskq1 = (q1 >= q_min) & (q1 <= q_max)

    q1_masked = q1[maskq1]
    I1_masked = I1[maskq1]
    I2_masked = I2[maskq1]

    # Compute the log intensity ratio
    logdI = np.log(I1_masked / I2_masked)

    # Map form_factor_name to an f2_initial if recognized
    f2_dictionary = {
        'guinier_ff': 0,
        'sphere_ff': 1/126,
        'gaussian_ff': -1/36
    }

    # Determine f2 parameter
    if form_factor_name == 'NaN':
        f2_free = True
        f2_initial = random.uniform(-1, 1)
        print('No form factor given; f2 is fitted as a free parameter.')
    elif form_factor_name in f2_dictionary:
        f2_free = False
        f2_initial = f2_dictionary[form_factor_name]
    else:
        f2_free = True
        print('Unrecognized form factor name; f2 is fitted as a free parameter.')

    # Update q_min and q_max from masked data
    q_min = np.min(q1_masked)
    q_max = np.max(q1_masked)

    # Ensure scaling factor is nonzero if initial guess was 0
    if A_initial == 0:
        A_initial = epsilon

    # Optionally perform Guinier estimation
    rg1 = guinier_approximation.estimate_Rg(q1_masked, I1_masked, q_min, q_max)
    rg2 = guinier_approximation.estimate_Rg(q1_masked, I2_masked, q_min, q_max)
    rg_guinier = np.mean([rg1, rg2])
    # print("auto Rg =", rg_guinier)

    if perform_guinier_estimation or auto_set_parameters:
        rg_initial = rg_guinier

    # If parameters are not free, fix their bounds to the initial value
    if not rg_free:
        rg_min = rg_initial
        rg_max = rg_initial
    if not f2_free:
        f2_min = f2_initial
        f2_max = f2_initial
    if not var_free:
        var_min = var_initial
        var_max = var_initial
    if not A_free:
        A_min = A_initial
        A_max = A_initial

    # Automatically set bounds for rg and A if requested
    if auto_set_parameters or auto_init_A:
        logdI_zero = logdI[0]
        # Calculate A_initial based on the 0th q-value using G_function(0, ...), assumed to be valid
        A_initial = logdI_zero / G_function_new(0, rg_initial, f2_initial, var_initial, 1)
    
    if auto_set_parameters:
        rg_min = max(epsilon, rg_initial * (1 - auto_rg_bound_percent))
        rg_max = rg_initial * (1 + auto_rg_bound_percent)  
        # Expand/minimize A around the new A_initial
        A_max = max(A_initial * (1 - auto_rg_bound_percent), A_initial * (1 + auto_rg_bound_percent))
        A_min = min(A_initial * (1 - auto_rg_bound_percent), A_initial * (1 + auto_rg_bound_percent))

    # Collect bounds
    lower_bounds = [rg_min, f2_min, var_min, A_min]
    upper_bounds = [rg_max, f2_max, var_max, A_max]

    # Adjust bounds to ensure they are valid (assumes adjust_bounds is defined elsewhere)
    lower_bounds, upper_bounds = adjust_bounds(lower_bounds, upper_bounds, epsilon)

    # Build initial guess array
    p0 = [rg_initial, f2_initial, var_initial, A_initial]
    # Fit logdI vs q using SciPy curve_fit
    try:
        popt, pcov = curve_fit(G_function_new, q1_masked, logdI,
                               p0=p0, bounds=(lower_bounds, upper_bounds),
                               maxfev=maxfev)
    except Exception as e:
        print("Error during curve_fit:", e)
        return None

    # Compute standard errors
    if pcov is not None:
        errors = np.sqrt(np.diag(pcov))
    else:
        errors = [None] * len(popt)

    # Organize results
    fit_results = {
        "optimal_parameters": {
            "rg_fit": popt[0],
            "rg_fit_error": errors[0],
            "f2_fit": popt[1],
            "f2_fit_error": errors[1],
            "var_fit": popt[2],
            "var_fit_error": errors[2],
            "A_fit": popt[3],
            "A_fit_error": errors[3],
            "Rg_guinier": rg_guinier
        },
        "covariance": pcov
    }

    # Calculate model values for the initial and final parameters
    G_initial = G_function_new(q1_masked, rg_initial, f2_initial, var_initial, A_initial)
    G_final = G_function_new(q1_masked, *popt)

    # Return fit dictionary and relevant arrays
    return fit_results, q1_masked, logdI, G_final, G_initial    

def plot_G_function_fits(q1_masked, logdI, G_final, G_initial, fit_results,
                         filename1=None, filename2=None):
    """
    Plot data (q1_masked vs. logdI) along with two curves:
      - G_final (the best fit)
      - G_initial (the initial guess)

    Parameters
    ----------
    q1_masked : ndarray
        The masked q-values used in the fit (1D array).
    logdI : ndarray
        The log intensity ratio, ln(I1/I2), used for plotting (1D array).
    G_final : ndarray
        The final fitted G-function values for each q in q1_masked.
    G_initial : ndarray
        The initial G-function values for each q in q1_masked.
    fit_results : dict
        A dictionary with keys "optimal_parameters" and "covariance", e.g.:

        fit_results= {
            "optimal_parameters": {
                "rg_fit": popt[0],
                "rg_fit_error": errors[0],
                "f2_fit": popt[1],
                "f2_fit_error": errors[1],
                "var_fit": popt[2],
                "var_fit_error": errors[2],
                "A_fit": popt[3],
                "A_fit_error": errors[3],
                "Rg_guinier": rg_guinier
            },
            "covariance": pcov
        }

    filename1 : str, optional
        Name/path of the first file, to be displayed in the annotation.
    filename2 : str, optional
        Name/path of the second file, to be displayed in the annotation.

    Returns
    -------
    None
        Displays a Matplotlib figure with the data, initial guess, and best fit.
    """

    # Extract the final fit parameters from the fit_results dictionary
    optimal_params = fit_results["optimal_parameters"]
    rg_fit = optimal_params["rg_fit"]       # e.g., popt[0]
    var_fit = optimal_params["var_fit"]     # e.g., popt[2]

    # Create the plot
    plt.figure(figsize=(8, 6))
    
    # Plot the raw data points
    plt.plot(q1_masked**2, logdI, 'bo', label='Data')

    # Plot the best fit
    plt.plot(q1_masked**2, G_final, 'r-', label='Best fit')

    # Plot the initial guess
    plt.plot(q1_masked**2, G_initial, 'k--', label='Initial parameters')

    # Label and title
    plt.xlabel('q^2')
    plt.ylabel('ln(I1/I2) (normalized)')
    plt.title('G_function Fitting')

    # Build annotation string
    annotation = ""
    if filename1 is not None:
        annotation += f"File 1: {filename1}\n"
    if filename2 is not None:
        annotation += f"File 2: {filename2}\n"

    # Here we assume "rg_fit" is the final Rg and "var_fit" is the final V
    annotation += f"Rg: {rg_fit:.3f}\n"
    annotation += f"V: {var_fit:.3f}\n"

    # Place annotation in the upper left corner of the plot
    plt.gca().text(
        0.05, 0.95, annotation,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    plt.legend()
    plt.show()