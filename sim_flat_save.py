import analytical_simulation_2d
import binning
import G_approximation
import matplotlib.pyplot as plt
import numpy as np
import save_and_load as csv_man

# *** Simulation parameters ***
'''
* Detector and source parameters

    -- px_number -- number of detector pixels [X, Y],
    -- px_size --  linear pixel size in mm
    -- wavelength -- source wavelength in  # nm

* "Sample" parameters

    -- form_factor_name -- name of the form factor from form_factor_methods:
        - guinier_ff #GUINIER FORM FACTOR
        - sphere_ff #SPHERICAL FORM FACTOR
        - gaussian_ff #GAUSSIAN CHAIN FORM FACTOR

    -- rg -- Rg in THE SAME UNITS AS WAVELENGTH
    -- variance
    -- sigma_x -- in pixels
    -- sigma_y -- in pixels
    -- sample_detector_distance -- in mm

Flattening parameters
    -- normalization_on: bool
        A flag that allows to choose if the flattening should be done with normalization
            True: with normalization
            False: without normalization
    -- return_unique : bool, optional
        If True, return unique radial values with averaged intensities (default is True).
    -- q_min : float, optional
        Minimum q value for filtering (only applied if q_max is also provided).
    -- q_max : float, optional
        Maximum q value for filtering
'''

# *** Binning parameters ***
'''
    -- bins_number ---
'''

# *** Fitting parameters ***
'''
Parameters for fitting the ln(I1/I2) vs q data with G function

Input parameters:

Arbitrary parameters:
    - q1, I1, q2, I2 -- two sets of Intensities vs q

Optional parameters:
    * form_factor_name
        If form_factor_name is given the algorithm will use known f2
        If form_factor_name is NOT given the algorithm will fit f2 as free parameter
    * f2
    -- f2_initial can be either defined by form_factor_name, or manually, or not defined 
    -- form_factor_name -- used to define f2 parameter value:  
            guinier_ff: Guinier form factorm, f2 == 0
            sphere_ff: for spherical particles, f2 == 1/126
            gaussian_ff: for gaussian chain, f2 == -1/36
    -- f2_free -- Default: True, if True, treated as a free parameter. If False -- fixed.
    -- f2_min, f2_max -- Default in fange [-1,1] 
    

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
    
    * A (scale factor of G function)
    -- A_initial -- avoid 0 values. If 0, will be changed to 1
    -- A_free -- Default: True
    -- A_min, A_max
    
    -- q_min, q_max -- Default: in full q range of the input data
    
    -- fitting_method -- selection of fitting methods from .lmfit library https://lmfit.github.io/lmfit-py/fitting.html. Default: 'leastsq' 
    
    -- save_to_log -- optional saving  of the fit results to log file, Default: False
    -- log_file_name -- Default: 'auto_log_file'
'''

### *** ENTER ALL SIMULATION AND FITTING PARAMETERS ***

## ** Enter parameters for the 1st I(q) data set **
simulation_parameters_1 = {
# Simulation parameters
# Detector and source parameters
    "px_number": [500, 500],
    "px_size": 0.075,  # mm
    "wavelength": 0.154,  # nm
# "Sample" parameters
    "form_factor_name": "guinier_ff",
    "rg": 2,  # in nm
    "variance": 0.01,
    "sigma_x": 0.01,  # in pixels
    "sigma_y": 0.01,  # in pixels
    "sample_detector_distance": 1500,  # in mm
# Flattening parameters
    "normalization_on": True,
    "return_unique": False,
    "q_min": 0.0001,
    "q_max": 0.5
}

## ** Enter parameters for the 2nd I(q) data set **
simulation_parameters_2 = {
# Simulation parameters
# Detector and source parameters
    "px_number": [500, 500],
    "px_size": 0.075,  # mm
    "wavelength": 0.154,  # nm
# "Sample" parameters
    "form_factor_name": "guinier_ff",
    "rg": 2,  # in nm
    "variance": 0.01,
    "sigma_x": 2,  # in pixels
    "sigma_y": 2,  # in pixels
    "sample_detector_distance": 1500,  # in mm
# Flattening parameters
    "normalization_on": True,
    "return_unique": False,
    "q_min": 0.00001,
    "q_max": 0.5
}

# Binning parameters
bins_number = 1000

# Fitting parameters

### * END OF INPUT PARAMETERS *

# Creates two sets of 1D flattened I(q) data based on params1 ans params2
# Returns 4 1D arrays: I1 vs q1 and I2 vs q2
q1, I1 = analytical_simulation_2d.single_analytical_simulation_flattened(simulation_parameters_1)
q2, I2 = analytical_simulation_2d.single_analytical_simulation_flattened(simulation_parameters_2)

# Bins the I(q) data sets and finds overlapping regions
# Returns 3 arrays: binned q (overlap between q1 and q2), binned I1 and binned I2
binned_data = binning.bin_and_match_saxs_data(q1, I1, q2, I2, bins_number)

# Assigns binned values
q1_binned = binned_data["q"].values
q2_binned = binned_data["q"].values
I1_binned = binned_data["I1"].values
I2_binned = binned_data["I2"].values

csv_man.save_q_I_to_csv (q1_binned,I1_binned,filename="I_q1_guinier_ff.csv")
csv_man.save_q_I_to_csv (q2_binned,I2_binned,filename="I_q2_guinier_ff.csv")

q_units = 'nm'
plt.figure(1)
plt.xscale('log')
plt.yscale('log')
plt.title('Initial data')
plt.plot(q1,I1, label=f'I1')
plt.plot(q2,I2*(np.max(I1)/np.max(I2)*1.01), label=f'I2')
plt.xlabel(f'initial ln(q, 1/{q_units})')
plt.ylabel('initial ln(I)')

plt.figure(2)
plt.xscale('log')
plt.yscale('log')
plt.title('Binned data')
plt.plot(q1_binned,I1_binned, label=f'binned I1')
plt.plot(q2_binned,I2_binned*(np.max(I1_binned)/np.max(I2_binned)*1.01), label=f'binned I2')
plt.xlabel(f'binned ln(q, 1/{q_units})')
plt.ylabel('binned ln(I)')

plt.show()



