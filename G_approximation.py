import numpy as np
#import lmfit
from typing import Optional
import random
import guinier_approximation
import matplotlib.pyplot as plt
import save_read_convert

from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# Defines G(q) fitting function
#List of f2 values based on the form factor
f2_dictionary={'guinier_ff': 0, 'sphere_ff': 1/126, 'gaussian_ff': -1/36}

# G(q) fitting function
# x is an argument (q)
# rg_fit -- Rg
# f2_fit -- f2 parameter
# var_fit -- variance
# A_fit -- scaling factor
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
    return (2/3) * rg2 * A_fit * (g0 + g1 + g2 + g3)

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

# def G_fit(q1, I1, q2, I2,
#           form_factor_name: Optional[str] = 'NaN',
#           f2_initial: Optional[float] = 0, f2_min: Optional[float] = -1, f2_max: Optional[float] = 1, f2_free: Optional[bool]=True,
#           q_min: Optional[float] = None, q_max: Optional[float] = None,
#           rg_initial: Optional[float] = 0, rg_min: Optional[float]=0, rg_max: Optional[float]=1e5, rg_free: Optional[bool]=True,
#           var_initial: Optional[float] = 0, var_min: Optional[float]=0, var_max: Optional[float]=1, var_free: Optional[bool]=True,
#           A_initial: Optional[float] = 0, A_min: Optional[float] = -1e38, A_max: Optional[float] = 1e38, A_free: Optional[bool]=True,
#           fitting_method: Optional[str] = 'leastsq',
#           perform_guinier_estimation: Optional[bool] = False,
#           save_to_log: Optional[bool] = False,
#           log_file_name: Optional[str] = 'auto_log_file',
#           plot_fitting_curve: Optional[bool] = False
#           ):

#     # Check if the q ranges of the input data sets are the same
#     if not np.array_equal(q1, q2): print('q ranges of the data sets do not match. Bin the data.')

#     # Masks the data based on selected q region
#     if q_min is None:
#         q_min = max(np.min(q1), np.min(q2))
#     if q_max is None:
#         q_max = min(np.max(q1), np.max(q2))
    
#     maskq1 = (q1 >= q_min) & (q1 <= q_max)
#     maskq2 = (q2 >= q_min) & (q2 <= q_max)
#     # elif (q_min is not None):
#     #     maskq1 = (q1 >= q_min)
#     #     maskq2 = (q2 >= q_min)
#     # elif (q_max is not None):
#     #     maskq1 = (q1 <= q_max)
#     #     maskq2 = (q2 <= q_max)
#     # else:
#     #     maskq1 = True
#     #     maskq2 = True

#     q1 = np.array(q1).flatten()
#     q2 = np.array(q2).flatten()
#     I1 = np.array(I1).flatten()
#     I2 = np.array(I2).flatten()


#     q1_masked = q1[maskq1]
#     q2_masked = q2[maskq2]
#     I1_masked = I1[maskq1]
#     I2_masked = I2[maskq2]

#     # Calculates logI = log(I1/I2)
#     logdI = np.log(I1_masked / I2_masked)
#     logdI = logdI / logdI[0]
#     if form_factor_name == 'NaN':
#         f2_free = True
#         f2_initial = random.uniform(-1, 1)
#         print('No form factor is given, f2 is fitted as a free parameter')
#     elif form_factor_name in f2_dictionary:
#         f2_free = False
#         f2_initial = f2_dictionary[form_factor_name]
#     else:
#         f2_free = True
#         #f2_initial = random.uniform(-1, 1)
#         print('Wrong form factor name, f2 is fitted as a free parameter')

#     q_min = np.min(q1_masked)
#     q_max = np.max(q1_masked)

#     if var_initial == 0:
#         var_initial = random.uniform(0, 1)

#     # if A_initial == 0:
#     #     A_initial += 1

#     if perform_guinier_estimation:
#         # Estimates Rgs from the 1st 1D data set
#         rg1 = guinier_approximation.estimate_Rg(q1_masked, I1_masked, q_min, q_max)
#         rg2 = guinier_approximation.estimate_Rg(q2_masked, I2_masked, q_min, q_max)
#         rg_initial = np.mean([rg1, rg2])

#     #Fitting logdI = log(I1/I2) with G function
#     #model=lmfit.Model(func=G_function, method=fitting_method)
#     model=lmfit.Model(func=G_function, method='differential_evolution')
    
#     parameters = lmfit.Parameters()
#     parameters.add("rg_fit", value=rg_initial, vary=rg_free, min=rg_min, max=rg_max)
#     parameters.add("f2_fit", value=f2_initial, vary=f2_free, min=f2_min, max=f2_max)
#     parameters.add("var_fit", value=var_initial, vary=var_free, min=var_min, max=var_max)
#     parameters.add("A_fit", value=A_initial, vary=A_free, min=A_min, max=A_max)

#     results=model.fit(logdI, params=parameters, x=q1_masked)

#     G_fit_results = results.fit_report()

#     # Save results to log if required
#     if save_to_log:
#         save_read_convert.save_fit_results_to_log(log_file_name, G_fit_results)

#     if plot_fitting_curve:
#         plt.figure(1)
#         plt.xscale('linear')
#         plt.yscale('linear')
#         plt.plot(q1_masked, logdI, label='Data')
#         plt.plot(q1_masked, results.best_fit, label='Best fit')
#         # Compute the G_function with the initial parameters
#         G_initial = G_function(q1_masked, rg_initial, f2_initial, var_initial, A_initial)
#         plt.plot(q1_masked, G_initial, 'k--', label='Initial parameters')
#         plt.title('Fitting curve')
#         plt.xlabel('q')
#         plt.ylabel('ln(I1/I2)')
#         plt.legend()
#         plt.show()

#     return G_fit_results



# Dictionary for f2 values based on form factor name
f2_dictionary = {'guinier_ff': 0, 'sphere_ff': 1/126, 'gaussian_ff': -1/36}
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

def print_fitted_results_with_errors(fit_results):
    params = fit_results["optimal_parameters"]
    cov = fit_results["covariance"]

    # If covariance is not None, compute standard errors as sqrt of the diagonal elements.
    if cov is not None:
        errors = np.sqrt(np.diag(cov))
    else:
        errors = [None] * len(params)

    # Specify the order of parameters expected by your model.
    param_order = ["rg_fit", "f2_fit", "var_fit", "A_fit"]

    print("Fitted Parameters (value ± error):")
    for i, param in enumerate(param_order):
        value = params.get(param, None)
        error = errors[i] if errors[i] is not None else "N/A"
        print(f"  {param}: {value:.6g} ± {error:.6g}" if isinstance(error, float) else f"  {param}: {value} ± {error}")

def G_fit(q1, I1, q2, I2,
          form_factor_name: Optional[str] = 'NaN',
          f2_initial: Optional[float] = 0, f2_min: Optional[float] = -1, f2_max: Optional[float] = 1, f2_free: Optional[bool] = True,
          q_min: Optional[float] = None, q_max: Optional[float] = None,
          rg_initial: Optional[float] = 0, rg_min: Optional[float] = 0, rg_max: Optional[float] = 1e5, rg_free: Optional[bool] = True,
          var_initial: Optional[float] = 0, var_min: Optional[float] = 0, var_max: Optional[float] = 1, var_free: Optional[bool] = True,
          A_initial: Optional[float] = 0, A_min: Optional[float] = -1e38, A_max: Optional[float] = 1e38, A_free: Optional[bool] = True,
          perform_guinier_estimation: Optional[bool] = False,
          plot_fitting_curve: Optional[bool] = False,
          maxfev: Optional[int] = 10000,
          auto_set_parameters: Optional[bool]=True,
          auto_rg_bound_percent: Optional [float] = .1
          ):
    """
    Fits the log intensity ratio data using the G_function model and SciPy's curve_fit.

    The function processes two data sets (q1, I1) and (q2, I2) to compute
    logdI = log(I1/I2) (normalized by its first value) and then fits it with the
    G_function model defined elsewhere.

    Parameters
    ----------
    q1, I1, q2, I2 : array_like
        Two sets of intensities vs. q.
    form_factor_name : str, optional
        Name of the form factor to determine f2. If 'NaN' or unrecognized,
        f2 is treated as a free parameter.
    f2_initial, f2_min, f2_max : float, optional
        Initial guess and bounds for f2 parameter.
    f2_free : bool, optional
        If True, f2 is free to vary; if False, it will be fixed.
    q_min, q_max : float, optional
        q-range to use. If None, the overlapping range of q1 and q2 is used.
    rg_initial, rg_min, rg_max : float, optional
        Initial guess and bounds for radius of gyration.
    rg_free : bool, optional
        If True, rg is free to vary.
    var_initial, var_min, var_max : float, optional
        Initial guess and bounds for variance.
    var_free : bool, optional
        If True, variance is free to vary.
    A_initial, A_min, A_max : float, optional
        Initial guess and bounds for the scaling factor A.
    A_free : bool, optional
        If True, A is free to vary.
    perform_guinier_estimation : bool, optional
        If True, use Guinier approximation to estimate rg from the data.
    plot_fitting_curve : bool, optional
        If True, plot the data, the best fit, and the initial guess curve.
    maxfev : int, optional
        Maximum number of function evaluations for curve_fit.
    auto_set_parameters : bool, optional
        Set Rg by Guinier, Rg bound by auto_rg_bound_percent around it, and A accourdingly   
    auto_rg_bound_percent: float, optional
        how much Rg can vary around its mean if auto_set_parameter is true
    
    Returns
    -------
    fit_results : dict
        A dictionary containing the optimal parameters and the covariance matrix.
    """
    epsilon=1e-14
    # Warn if q ranges differ
    if not np.array_equal(q1, q2):
        print('q ranges of the data sets do not match. Consider binning the data.')

    # Determine common q-range if not provided
    if q_min is None:
        q_min = max(np.min(q1), np.min(q2))
    if q_max is None:
        q_max = min(np.max(q1), np.max(q2))

    # Create masks for q-range selection
    maskq1 = (q1 >= q_min) & (q1 <= q_max)
    maskq2 = (q2 >= q_min) & (q2 <= q_max)

    # Ensure data are 1D arrays
    q1 = np.array(q1).flatten()
    q2 = np.array(q2).flatten()
    I1 = np.array(I1).flatten()
    I2 = np.array(I2).flatten()

    q1_masked = q1[maskq1]
    q2_masked = q2[maskq2]
    I1_masked = I1[maskq1]
    I2_masked = I2[maskq2]

    # Compute normalized log intensity ratio
    logdI = np.log(I1_masked / I2_masked)
    #logdI = logdI / logdI[0]

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

    # Update q range from masked data
    q_min = np.min(q1_masked)
    q_max = np.max(q1_masked)
  
    
    # Ensure scaling factor is nonzero
    if A_initial == 0:
        A_initial = 1e-8

     

    # Perform Guinier estimation if requested
    if (perform_guinier_estimation | auto_set_parameters):
        rg1 = guinier_approximation.estimate_Rg(q1_masked, I1_masked, q_min, q_max)
        rg2 = guinier_approximation.estimate_Rg(q2_masked, I2_masked, q_min, q_max)
        rg_initial = np.mean([rg1, rg2])

  
    
   


    # For fixed parameters, set lower and upper bounds equal to the initial guess.
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
    logdI_zero = logdI[0]
    print ("ln(I1(0)/I2(0))=",logdI_zero)
    if auto_set_parameters:
       
        rg_min = np.max([epsilon,rg_initial*(1-auto_rg_bound_percent)])
        rg_max = rg_initial*(1+auto_rg_bound_percent)
        A_initial = logdI_zero / G_function(0, rg_initial, f2_initial, var_initial, 1)
        A_max =  np.max([epsilon,logdI_zero / G_function(0, rg_max, f2_initial, var_max, 1)])
        A_min =  logdI_zero / G_function(0, rg_min, f2_initial, var_min, 1)
        print ("ln(I1(0)/I2(0))=",logdI_zero)
        
    lower_bounds = [rg_min, f2_min, var_min, A_min]
    upper_bounds = [rg_max, f2_max, var_max, A_max]
    
    # Adjust bounds to ensure each lower bound is strictly less than its upper bound.
    lower_bounds, upper_bounds = adjust_bounds(lower_bounds, upper_bounds, epsilon)
    
    # Build initial guess vector and bounds
    p0 = [rg_initial, f2_initial, var_initial, A_initial]
    
    # Perform curve fitting using SciPy's curve_fit
    try:
        popt, pcov = curve_fit(G_function, q1_masked, logdI, p0=p0, bounds=(lower_bounds, upper_bounds), maxfev=maxfev)
    except Exception as e:
        print("Error during curve_fit:", e)
        return None

    # Prepare a simple fit report
    fit_results = {
        "optimal_parameters": {
            "rg_fit": popt[0],
            "f2_fit": popt[1],
            "var_fit": popt[2],
            "A_fit": popt[3]
        },
        "covariance": pcov
    }

    # Optionally save results to a log file
    # (Assumes save_read_convert.save_fit_results_to_log is implemented)
    # Uncomment the next lines if you want to save the fit report.
    # report_str = "\n".join(f"{key}: {val}" for key, val in fit_results["optimal_parameters"].items())
    # save_read_convert.save_fit_results_to_log("auto_log_file", report_str)

    # Optionally plot the fitting curve
    if plot_fitting_curve:
        plt.figure(figsize=(8, 6))
        plt.plot(q1_masked, logdI, 'bo', label='Data')
        plt.plot(q1_masked, G_function(q1_masked, *popt), 'r-', label='Best fit')
        # Compute and plot the curve using the initial parameters
       # G_initial = G_function(q1_masked, rg_initial, f2_initial, var_initial, A_initial)
       # plt.plot(q1_masked, G_initial, 'k--', label='Initial parameters')
        plt.xlabel('q')
        plt.ylabel('ln(I1/I2) (normalized)')
        plt.title('G_function Fitting')
        plt.legend()
        plt.show()

    return fit_results