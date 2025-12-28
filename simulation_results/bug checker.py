# here we'll check for bug in the code. First the data creation.
# we want to see we can graph an accruate Boltzmann distribution
# that we mange to graph a I(q) plot for a spherical particle
# and then a graph that marks eleven points on a boltz' dis, and another graph with the eleven spherical I(q) plots
# it gives + the sum of it, compared to our own graphing.
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from matplotlib.colors import Normalize, BoundaryNorm
import matplotlib.cm as cm
# so, graph boltzman
from distribution_functions import Boltzmann_dis

y = lambda x: Boltzmann_dis(x, 0.1, 1)

# Define your range (centered around the mean of 1)
x_vals = np.linspace(0.5, 1.5, 500)
y_vals = [y(x) for x in x_vals]

plt.figure(figsize=(8, 5))
plt.plot(x_vals, y_vals, label='Boltzmann Distribution', color='blue', lw=2)

# Mark the Expectation Value (Mean = 1)
plt.axvline(1, color='red', linestyle='--', label='Expectation Value (μ = 1)')

# Represent the Standard Deviation (0.1)
# We shade the area between μ - σ and μ + σ
plt.axvspan(1 - 0.1, 1 + 0.1, alpha=0.2, color='gray', label='Std Dev (σ = 0.1)')

plt.xlabel('Particle Dimension (R in nm)')
plt.ylabel('Probability Density (1/nm)')
plt.title('Boltzmann Distribution Analysis')
plt.legend()
plt.show()

# let's also make sure the integral is 1
# Define your integration range (0 to infinity for Boltzmann)
# Using 10 as an upper bound is safe if mean=1 and std=0.1
integral_value, error = quad(y, 0, 10)

print(f"The integral under the curve is: {integral_value}")
if abs(1 - integral_value) < 1e-4:
    print("The distribution is correctly normalized.")
else:
    print("Warning: The distribution is not normalized to 1.")
# we got 0.9999, we're cool

# graph spherical

def I_spherical_single(qR):
    """
    Computes intensity I as a function of the dimensionless parameter qR.
    """
    # Handle qR=0 to avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        # The form factor F(qR)
        bes_val = 3 * (np.sin(qR) - qR * np.cos(qR)) / (qR ** 3)
        # compare with our proper functions bes_val = 3 * (np.sin(qR) - qR * np.cos(qR)) / (qR ** 3)
        # Limit of 1.0 for qR -> 0
        bes = np.where(qR < 1e-9, 1.0, bes_val)

    return bes ** 2


# Define a range for qR (dimensionless)
# 0 to 15 is usually enough to see the first three minima
qr_axis = np.linspace(0, 15, 1000)
intensity = I_spherical_single(qr_axis)

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(qr_axis, intensity, label='Sphere Form Factor $|I(qR)|^2$', color='red', lw=2)

# Log scale is still recommended to see the oscillations clearly
plt.yscale('log')

# Adding reference lines for the known minima positions
minima = [4.493, 7.725, 10.904]
for m in minima:
    plt.axvline(m, color='black', linestyle=':', alpha=0.6)

plt.title('Universal Scattering Curve: $I$ vs. $qR$')
plt.xlabel('$qR$')
plt.ylabel('$I(qR)$ (Normalized)')
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.legend()
plt.show()

# combine

# first let's make sure we manage to TAKE 3 dots
# 1. Define the curve
# Assuming y(x) is your Boltzmann lambda: y = lambda x: Boltzmann_dis(x, 0.1, 1)
x_curve = np.linspace(0.5, 1.5, 500)
y_curve = [y(x) for x in x_curve]

# Mark the 11 points as diamonds ('D')
x_11 = np.linspace(0.8, 1.2, 11)
y_11 = y(x_11)

# 3. Create a color array for 11 distinct colors
colors = plt.get_cmap('rainbow')(np.linspace(0, 1, 11))

# 4. Graphing
plt.figure(figsize=(10, 6))

# Plot the underlying distribution
plt.plot(x_curve, y_curve, color='gray', alpha=0.3, label='Boltzmann PDF')

# Plot the 11 points as colored diamonds
plt.scatter(x_11, y_11, c=colors, marker='D', s=80, edgecolors='black', zorder=5, label='11 Sampling Points')

plt.title("Boltzmann Distribution: 11 Discrete Sampling Points")
plt.xlabel("Radius (nm)")
plt.ylabel("Probability Density")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.show()

#compare
# First, let's compare I's without wights:
# 1. Setup the q range
q_vals = np.logspace(-1, 1, 1000)  # Using logspace for better log-axis sampling

plt.figure(figsize=(10, 6))

# 2. Create DISCRETE boundaries for the colorbar
# We want the boundaries to be halfway between our x_11 points
boundaries = np.zeros(len(x_11) + 1)
boundaries[1:-1] = (x_11[:-1] + x_11[1:]) / 2
boundaries[0] = x_11[0] - (x_11[1] - x_11[0]) / 2  # Padding first edge
boundaries[-1] = x_11[-1] + (x_11[-1] - x_11[-2]) / 2  # Padding last edge

# 3. Setup Discrete Mapping
# 'ncolors' should match the number of points (11)
cmap = plt.get_cmap('rainbow', len(x_11))
norm = BoundaryNorm(boundaries, cmap.N)
mappable = cm.ScalarMappable(norm=norm, cmap=cmap)

# 4. Plot the 11 individual curves
for i, R in enumerate(x_11):
    qR = q_vals * R
    I_single = I_spherical_single(qR)

    # Use the discrete cmap to get the color
    plt.plot(q_vals, I_single, color=cmap(i), alpha=0.7, lw=1.5)

# 5. Add the Discrete Colorbar
cbar = plt.colorbar(mappable, ax=plt.gca(), ticks=x_11)
cbar.set_label('Radius $R$ (nm)', rotation=270, labelpad=15)
cbar.ax.set_yticklabels([f'{val:.2f}' for val in x_11])  # Label with actual R values

# 6. Styling
plt.xscale('log')
plt.yscale('log')
plt.title('11 Discrete Spherical $I(q)$ Plots')
plt.xlabel('$q$ (nm$^{-1}$)')
plt.ylabel('$I(q)$')
plt.grid(True, which="both", ls="-", alpha=0.2)

plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from matplotlib.colors import Normalize, BoundaryNorm
import matplotlib.cm as cm

# --- 1. Physics Function: Now returning F instead of I ---
def F_spherical_single(qR):
    """
    Computes the Form Factor amplitude F(qR).
    Note: I(qR) = F(qR)**2
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        bes_val = 3 * (np.sin(qR) - qR * np.cos(qR)) / (qR ** 3)
        # Limit is 1.0 for qR -> 0
        F = np.where(qR < 1e-9, 1.0, bes_val)
    return F

# --- 2. Universal Curve (F vs qR) ---
qr_axis = np.linspace(0, 15, 1000)
F_vals = F_spherical_single(qr_axis)

plt.figure(figsize=(8, 5))
plt.plot(qr_axis, F_vals, label='Sphere Form Factor $F(qR)$', color='red', lw=2)
plt.axhline(0, color='black', lw=1) # Reference line for zero-crossings

# Adding minima (where F is negative/positive peaks)
minima = [4.493, 7.725, 10.904]
for m in minima:
    plt.axvline(m, color='black', linestyle=':', alpha=0.4)

plt.title('Universal Scattering: Form Factor Amplitude $F$ vs. $qR$')
plt.xlabel('$qR$')
plt.ylabel('$F(qR)$')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# --- 3. Comparison of 11 Radii (F vs q) ---
q_vals = np.linspace(0.1, 10, 1000)

plt.figure(figsize=(10, 6))

# Use your existing x_11 and discrete cmap setup
boundaries = np.zeros(len(x_11) + 1)
boundaries[1:-1] = (x_11[:-1] + x_11[1:]) / 2
boundaries[0] = x_11[0] - (x_11[1] - x_11[0]) / 2
boundaries[-1] = x_11[-1] + (x_11[-1] - x_11[-2]) / 2

cmap = plt.get_cmap('rainbow', len(x_11))
norm = BoundaryNorm(boundaries, cmap.N)
mappable = cm.ScalarMappable(norm=norm, cmap=cmap)

for i, R in enumerate(x_11):
    qR = q_vals * R
    F_single = F_spherical_single(qR) # Calculating Amplitude F
    plt.plot(q_vals, F_single, color=cmap(i), alpha=0.7, lw=1.5)

plt.axhline(0, color='black', lw=1.5, linestyle='-')
cbar = plt.colorbar(mappable, ax=plt.gca(), ticks=x_11)
cbar.set_label('Radius $R$ (nm)', rotation=270, labelpad=15)
cbar.ax.set_yticklabels([f'{val:.2f}' for val in x_11])

plt.title('11 Discrete Spherical $F(q)$ Amplitudes')
plt.xlabel('$q$ (nm$^{-1}$)')
plt.ylabel('$F(q)$')
plt.grid(True, alpha=0.2)
plt.show()

# 1. Define Parameters
rg_avg = 1.0  # Average Radius of Gyration
sigma = 0.1  # From your Boltzmann(x, 0.1, 1)
scaling_factor = rg_avg * (1 + sigma ** 2)

# 2. Setup q range (nm^-1)
q_vals = np.linspace(0.1, 15, 1000)

# 3. Setup the Discrete Color Mapping (11 points)
x_11 = np.linspace(0.8, 1.2, 11)
boundaries = np.zeros(len(x_11) + 1)
boundaries[1:-1] = (x_11[:-1] + x_11[1:]) / 2
boundaries[0] = x_11[0] - (x_11[1] - x_11[0]) / 2
boundaries[-1] = x_11[-1] + (x_11[-1] - x_11[-2]) / 2

cmap = plt.get_cmap('rainbow', len(x_11))
norm = BoundaryNorm(boundaries, cmap.N)
mappable = cm.ScalarMappable(norm=norm, cmap=cmap)

plt.figure(figsize=(10, 6))

# 4. Loop through the 11 radii
for i, R_fraction in enumerate(x_11):
    # Convert fraction to actual physical radius R
    # Using your formula: R = (rg * r_fraction) * sqrt(5/3)
    R_physical = (rg_avg * R_fraction) * np.sqrt(5 / 3)

    # Calculate F based on physical q*R
    qR = q_vals * R_physical
    F_single = F_spherical_single(qR)

    # GRAPHING: x-axis is now q * scaling_factor
    x_axis_scaled = q_vals * scaling_factor
    plt.plot(x_axis_scaled, F_single, color=cmap(i), alpha=0.7, lw=1.5)

# 5. Formatting
plt.axhline(0, color='black', lw=1)
cbar = plt.colorbar(mappable, ax=plt.gca(), ticks=x_11)
cbar.set_label('Radius Fraction', rotation=270, labelpad=15)

plt.title(f'Form Factor Amplitudes scaled by $q \cdot R_g(1+\sigma^2)$')
plt.xlabel(r'$q \cdot R_g(1+\sigma^2)$ (Dimensionless)')
plt.ylabel('$F(q)$')
plt.grid(True, alpha=0.2)

plt.show()

# 1. Define Parameters
rg_avg = 1.0
sigma = 0.1
scaling_factor = rg_avg * (1 + sigma ** 2)

# 2. Setup q range
q_vals = np.linspace(0.1, 15, 1000)
x_axis_scaled = q_vals * scaling_factor

# 3. Use your Boltzmann function to get weights
# x_11 are the 11 radii fractions we chose (0.8 to 1.2)
# y_11 are the weights from the distribution
weights_11 = np.array([y(r) for r in x_11])

plt.figure(figsize=(10, 6))

# 4. Loop through and plot Weighted Form Factor: F(q) * P(R)
for i, R_fraction in enumerate(x_11):
    # Physical Radius calculation
    R_physical = (rg_avg * R_fraction) * np.sqrt(5 / 3)

    # Calculate Form Factor Amplitude
    qR = q_vals * R_physical
    F_single = F_spherical_single(qR)

    # Apply the weight from Boltzmann
    weighted_F = F_single * weights_11[i]

    # Plot using the discrete rainbow color
    plt.plot(x_axis_scaled, weighted_F, color=cmap(i), alpha=0.8, lw=1.5)

# 5. Styling
plt.axhline(0, color='black', lw=1.5)
cbar = plt.colorbar(mappable, ax=plt.gca(), ticks=x_11)
cbar.set_label('Radius Fraction $R/R_g$', rotation=270, labelpad=15)

plt.title(r'Weighted Form Factor Amplitudes: $F(q) \cdot P(R)$')
plt.xlabel(r'$q \cdot R_g(1+\sigma^2)$')
plt.ylabel(r'Contribution to Total $F$')
plt.grid(True, alpha=0.2)

plt.show()


# Mocking the Boltzmann distribution for demonstration
def Boltzmann_dis(x, sigma, mu):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


# Parameters
rg_avg = 1.0
sigma = 0.1
scaling_factor = rg_avg * (1 + sigma ** 2)
y_func = lambda x: Boltzmann_dis(x, sigma, rg_avg)

# q range
q_vals = np.linspace(0.01, 15, 1000)
x_axis_scaled = q_vals * scaling_factor

# 11 sampling points
x_11 = np.linspace(0.8, 1.2, 11)
weights_11 = np.array([y_func(x) for x in x_11])


# Physics Function
def F_spherical_single(qR):
    with np.errstate(divide='ignore', invalid='ignore'):
        bes_val = 3 * (np.sin(qR) - qR * np.cos(qR)) / (qR ** 3)
        F = np.where(qR < 1e-9, 1.0, bes_val)
    return F


# Setup Plot
plt.figure(figsize=(12, 7))

# Discrete mapping for colorbar
boundaries = np.zeros(len(x_11) + 1)
boundaries[1:-1] = (x_11[:-1] + x_11[1:]) / 2
boundaries[0] = x_11[0] - (x_11[1] - x_11[0]) / 2
boundaries[-1] = x_11[-1] + (x_11[-1] - x_11[-2]) / 2

cmap = plt.get_cmap('rainbow', len(x_11))
norm = BoundaryNorm(boundaries, cmap.N)
mappable = cm.ScalarMappable(norm=norm, cmap=cmap)

# To store the sum
total_weighted_F = np.zeros_like(q_vals)

for i, R_fraction in enumerate(x_11):
    R_physical = (rg_avg * R_fraction) * np.sqrt(5 / 3)
    qR = q_vals * R_physical
    F_single = F_spherical_single(qR)

    # DEBUG: Check if weight is 0
    weight = weights_11[i]
    if i == 5:  # Check the middle point (the peak)
        print(f"DEBUG: Radius={R_physical:.2f}, Weight={weight:.4f}, F_start={F_single[0]:.2f}")

    weighted_F = F_single * weight
    total_weighted_F += weighted_F

    plt.plot(x_axis_scaled, weighted_F, color=cmap(i), alpha=0.5, lw=1)

# Plot the SUM in BLACK
# 2. Plot your data
plt.plot(x_axis_scaled, total_weighted_F, color='black', lw=4, label='Total Summed F', zorder=10)

# 3. SET THE RANGE based on the maximum value of your sum
y_max = np.max(total_weighted_F)
y_min = np.min(total_weighted_F)

# Add a 10% buffer so the line doesn't touch the top of the box
plt.ylim(y_min - 0.1*y_max, y_max * 1.1)

# 4. Use a different style for the zero-axis so it's not black
plt.axhline(0, color='gray', linestyle=':', alpha=0.5, zorder=1)

plt.legend(loc='upper right')
plt.show()

# 1. Calculate the Squared Intensity
total_I_coherent = total_weighted_F**2

plt.figure(figsize=(10, 6))

# 2. Plot the result
plt.plot(x_axis_scaled, total_I_coherent, color='black', lw=3, label='Coherent Intensity $| \sum F \cdot P |^2$')

# 3. Styling - Intensity is usually viewed on a Log scale
plt.yscale('log')
# If you want to see the first decade of q, log x is also good
plt.xscale('log')

plt.title('Final Result: Coherent Intensity of the Ensemble')
plt.xlabel(r'$q \cdot R_g(1+\sigma^2)$')
plt.ylabel('Intensity (Arbitrary Units)')
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.legend()

plt.show()