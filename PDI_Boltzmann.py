import numpy as np
import matplotlib.pyplot as plt

def second_moment(sigma):
    return 1 + sigma**2

def sixth_moment(sigma):
    return 1 + 15*sigma**2 + 90*sigma**4 + 90*sigma**6

def eight_moment(sigma):
    return 1 + 28*sigma**2 + 420*sigma**4 + 2520*sigma**6 + 2520*sigma**8

def PDI_func(sigma):
    return (second_moment(sigma) * eight_moment(sigma)**2) / (sixth_moment(sigma)**3)

# ----------------------------------------------------------
# 1. Generate sigma and PDI values
# ----------------------------------------------------------
sigma = np.linspace(0, 1.5, 100)
PDI_values = PDI_func(sigma)

# ----------------------------------------------------------
# 2. Fit a n th-degree polynomial: sigma ≈ poly(PDI)
# ----------------------------------------------------------
degree = 10
coeffs = np.polyfit(PDI_values, sigma, degree)
poly = np.poly1d(coeffs)

# Print the polynomial
print("Fitted polynomial σ(PDI):")
print(poly)

# ----------------------------------------------------------
# 3. Plot data + fit
# ----------------------------------------------------------
PDI_fit = np.linspace(PDI_values.min(), PDI_values.max(), 500)
sigma_fit = poly(PDI_fit)

plt.plot(PDI_values, sigma, 'o', markersize=4, label='Data')
plt.plot(PDI_fit, sigma_fit, '-', label=f'{degree}th-degree fit')

plt.xlabel("PDI")
plt.ylabel("p")
plt.legend()
plt.show()
