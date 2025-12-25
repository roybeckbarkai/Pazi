import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erf
import matplotlib.pyplot as plt

# Custom imports for your specific environment
from data_creation_spherical import Scatter2D_spherical
from distribution_functions import Boltzmann_dis

# 1. The Core Model
def guinier_fit_ext(q_sq, G, Rg, B):
    q = np.sqrt(q_sq)
    # errstate handles the division by zero at q=0
    with np.errstate(divide='ignore', invalid='ignore'):
        guinier = G * np.exp(-q_sq * Rg**2 / 3.0)
        porod = B * (q**-4) * (erf(q * Rg / np.sqrt(6))**12)
    return np.nan_to_num(guinier + porod)

# 2. Generate Data
qx, qy, I, _, _ = Scatter2D_spherical(
    rg=2, peak_photon_density=10**11, pixel_count_along_detector=1000,
    distance_to_detector=150, wavelength=0.15, detector_length=7.0,
    smear_dims=(25,25), amount_of_radii_fractions=11,
    radii_fraction_difference=0.08,
    distribution_function=lambda x: Boltzmann_dis(x, 0.1, 1)
)

# 3. Data Flattening
q_sq_data = (qx**2 + qy**2).flatten()
I_data = I.flatten()

# 4. Perform Fit (using all 1,000,000 points)
# p0: [G_guess, Rg_guess, B_guess]
p_opt, _ = curve_fit(guinier_fit_ext, q_sq_data, I_data, p0=[np.max(I_data), 5.0, 1e2])
G_f, Rg_f, B_f = p_opt

# 5. Physics Calculations (PDI and p-factor)
PDI = (50 / 81) * ((B_f * (Rg_f**4)) / G_f)
p_coeffs = [2.103e-6, -8.236e-5, 0.001323, 0.01087, 0.0429, -0.007363, -0.6952, 3.146, -6.594, 7.004, -2.863]
p_val = np.poly1d(p_coeffs)(PDI)

print(f"Analysis Results:\n- Rg: {Rg_f:.4f}\n- PDI: {PDI:.4f}\n- p: {p_val:.4f}")

# 6. Simplified Plotting
plt.figure(figsize=(8, 6))
plt.scatter(q_sq_data * Rg_f**2, np.log(I_data), s=0.1, color='gray', alpha=0.05, label="Raw Data")

# Create a smooth line for the fit visualization
# q_plot = np.linspace(q_sq_data.min(), q_sq_data.max(), 500)
# plt.plot(q_plot, guinier_fit_ext(q_plot, *p_opt), 'r-', lw=2, label="Guinier-Porod Fit")


plt.xlabel("$q^2$")
plt.ylabel("Intensity")
plt.title(f"Fit Results: $R_g$={Rg_f:.2f}, $p$={p_val:.3f}")
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.show()