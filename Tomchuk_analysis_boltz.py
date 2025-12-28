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

# 5. Physics Constants for Scaling
# These must match your Scatter2D_spherical input
rg_input = 2.0
sigma = 0.1
scaling_factor = rg_input * (1 + sigma**2)

# 6. Scaled Plotting
plt.figure(figsize=(10, 6))

# Calculate the new x-axis: q * Rg * (1 + sigma^2)
# We take sqrt of q_sq_data to get q
q_linear = np.sqrt(q_sq_data)
x_scaled = q_linear * scaling_factor

# Scatter the raw data
plt.scatter(x_scaled, np.log(I_data), s=0.1, color='gray', alpha=0.05, label="Raw Data (Scaled)")

# Optional: Add the fit line on the same scaled axis
# We create a q_range for the line
q_plot_linear = np.linspace(q_linear.min(), q_linear.max(), 500)
I_fit = guinier_fit_ext(q_plot_linear**2, *p_opt)

plt.plot(q_plot_linear * scaling_factor, np.log(I_fit), 'r-', lw=2, label="Guinier-Porod Fit")

# Labeling
plt.xlabel(r"$q \cdot R_g(1+\sigma^2)$ (Scaled Dimensionless Axis)")
plt.ylabel("$\ln(Intensity)$")
plt.title(f"Scaled Fit: $R_g$={Rg_f:.2f}, $p$={p_val:.3f}")
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.show()