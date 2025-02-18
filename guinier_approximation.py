import numpy as np
from scipy.optimize import curve_fit

# Step 1: Define the linear Guinier function
def guinier_func(q_squared, ln_I0, Rg_squared):
    """
    Linear form of the Guinier equation:
    ln(I(q)) = ln(I0) - (Rg^2 / 3) * q^2
    """
    return ln_I0 - (Rg_squared / 3) * q_squared

# Step 2: Find the Guinier region and estimate Rg
def estimate_Rg(q, intensity, q_min, q_max):
    """
    Estimate the radius of gyration (Rg) using the Guinier approximation.

    Parameters:
    - q: Array of q values (scattering vector magnitudes)
    - intensity: Array of scattering intensities corresponding to q
    - q_max_threshold: Threshold for the Guinier region (small q)

    Returns:
    - Rg: Estimated radius of gyration
    - ln_I0: Extrapolated ln(I(0))
    """
    # Filter the data to find the Guinier region (q < q_max_threshold)
    mask = (q>=q_min) & (q <= q_max)
    intensity_guinier = intensity[mask]

    # Take the natural logarithm of the intensity
    ln_intensity = np.log(intensity_guinier)
    q_squared = q[mask] ** 2

    # Perform a linear fit using scipy.optimize.curve_fit
    popt, pcov = curve_fit(guinier_func, q_squared, ln_intensity)
    ln_I0, Rg_squared = popt

    # Calculate Rg
    rg_est = np.sqrt(Rg_squared)

    # Check if q * Rg < 1.3 for all q in the Guinier region
    if np.any(q * rg_est >= 1.3):
        print("Warning: Some q values in the Guinier region exceed the validity limit (q * Rg < 1.3).")
    else:
        print("All q values in the Guinier region satisfy q * Rg < 1.3.")

    return rg_est

'''
# Step 3: Example usage
# Simulated example data
q = np.linspace(0.01, 0.3, 50)  # q values from 0.01 to 0.3
Rg_true = 20  # True radius of gyration in Å
I0_true = 100  # True intensity at q=0
intensity = I0_true * np.exp(-(Rg_true**2 / 3) * q**2) + np.random.normal(0, 0.5, len(q))  # Add noise

# Estimate Rg using the function
Rg_est, ln_I0_est = estimate_Rg(q, intensity, q_max_threshold=0.15)
'''