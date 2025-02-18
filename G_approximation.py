import numpy as np
import lmfit
from typing import Optional
import random
import guinier_approximation
import matplotlib.pyplot as plt
import save_read_convert

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
    g0=-(1 + var_fit)
    g1= (2 / 3 - 18 * f2_fit + (16 / 3 - 108 * f2_fit) * var_fit) * x ** 2 * rg_fit ** 2
    g2= (8 * f2_fit + (176 * f2_fit - 16 / 9) * var_fit) * x ** 4 * rg_fit ** 4
    g3= (24 * f2_fit ** 2 + (960 * f2_fit ** 2 - 128 / 3 * f2_fit) * var_fit) * x ** 6 * rg_fit ** 6
    return 2/3*rg_fit*2*(g0 + g1 + g2 + g3) * A_fit

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

def G_fit(q1, I1, q2, I2,
          form_factor_name: Optional[str] = 'NaN',
          f2_initial: Optional[float] = 0, f2_min: Optional[float] = -1, f2_max: Optional[float] = 1, f2_free: Optional[bool]=True,
          q_min: Optional[float] = None, q_max: Optional[float] = None,
          rg_initial: Optional[float] = 0, rg_min: Optional[float]=0, rg_max: Optional[float]=1e5, rg_free: Optional[bool]=True,
          var_initial: Optional[float] = 0, var_min: Optional[float]=0, var_max: Optional[float]=1, var_free: Optional[bool]=True,
          A_initial: Optional[float] = 0, A_min: Optional[float] = -1e38, A_max: Optional[float] = 1e38, A_free: Optional[bool]=True,
          fitting_method: Optional[str] = 'leastsq',
          perform_guinier_estimation: Optional[bool] = False,
          save_to_log: Optional[bool] = False,
          log_file_name: Optional[str] = 'auto_log_file',
          plot_fitting_curve: Optional[bool] = False
          ):

    # Check if the q ranges of the input data sets are the same
    if not np.array_equal(q1, q2): print('q ranges of the data sets do not match. Bin the data.')

    # Masks the data based on selected q region
    if (q_min is not None) and (q_max is not None):
        maskq1 = (q1 >= q_min) & (q1 <= q_max)
        maskq2 = (q2 >= q_min) & (q2 <= q_max)
    elif (q_min is not None):
        maskq1 = (q1 >= q_min)
        maskq2 = (q2 >= q_min)
    elif (q_max is not None):
        maskq1 = (q1 <= q_max)
        maskq2 = (q2 <= q_max)
    else:
        maskq1 = True
        maskq2 = True

    q1_masked = q1[maskq1]
    q2_masked = q2[maskq2]
    I1_masked = I1[maskq1]
    I2_masked = I2[maskq2]

    # Calculates logI = log(I1/I2)
    logdI = np.log(I1_masked / I2_masked)

    if form_factor_name == 'NaN':
        f2_free = True
        f2_initial = random.uniform(-1, 1)
        print('No form factor is given, f2 is fitted as a free parameter')
    elif form_factor_name in f2_dictionary:
        f2_free = False
        f2_initial = f2_dictionary[form_factor_name]
    else:
        f2_free = True
        #f2_initial = random.uniform(-1, 1)
        print('Wrong form factor name, f2 is fitted as a free parameter')

    q_min = np.min(q1_masked)
    q_max = np.max(q1_masked)

    if var_initial == 0:
        var_initial = random.uniform(0, 1)

    if A_initial == 0:
        A_initial += 1

    if perform_guinier_estimation:
        # Estimates Rgs from the 1st 1D data set
        rg1 = guinier_approximation.estimate_Rg(q1_masked, I1_masked, q_min, q_max)
        rg2 = guinier_approximation.estimate_Rg(q2_masked, I2_masked, q_min, q_max)
        rg_initial = np.mean(rg1, rg2)

    #Fitting logdI = log(I1/I2) with G function
    model=lmfit.Model(func=G_function, method=fitting_method)
    parameters = lmfit.Parameters()
    parameters.add("rg_fit", value=rg_initial, vary=rg_free, min=rg_min, max=rg_max)
    parameters.add("f2_fit", value=f2_initial, vary=f2_free, min=f2_min, max=f2_max)
    parameters.add("var_fit", value=var_initial, vary=var_free, min=var_min, max=var_max)
    parameters.add("A_fit", value=A_initial, vary=A_free, min=A_min, max=A_max)

    results=model.fit(logdI, params=parameters, x=q1_masked)

    G_fit_results = results.fit_report()

    # Save results to log if required
    if save_to_log:
        save_read_convert.save_fit_results_to_log(log_file_name, G_fit_results)

    if plot_fitting_curve == True:
        plt.figure(1)
        plt.xscale('linear')
        plt.yscale('linear')
        plt.plot(q1_masked, logdI)
        #plt.plot(q1_masked, results.best_fit)
        #results.plot_fit(show_init=True)
        plt.title('Fitting curve')
        plt.xlabel('q')
        plt.ylabel('ln(I1/I2)')
        plt.show()

    return G_fit_results


