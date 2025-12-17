import matplotlib.pyplot as plt
import numpy as np

from distribution_functions import Boltzmann_dis
from data_creation_spherical import Scatter2D_spherical

def example_boltz(x):
    y = Boltzmann_dis(x, sigma=0.1, mean=1)
    return y

# attempt at a relevant smearing kernel
def generate_elliptical_kernel(size=51, sigma_x=10.0, sigma_y=2.0):
    """
    Creates a normalized 2D Gaussian kernel to stretch images into ellipses.
    sigma_x > sigma_y stretches the image horizontally.
    sigma_y > sigma_x stretches the image vertically.
    """
    # Create a coordinate grid centered at (0,0)
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    xx, yy = np.meshgrid(ax, ax)

    # Calculate the 2D Gaussian
    # Formula: exp(-(x^2/(2*sigma_x^2) + y^2/(2*sigma_y^2)))
    kernel = np.exp(-(xx ** 2 / (2 * sigma_x ** 2) + yy ** 2 / (2 * sigma_y ** 2)))

    # Normalize so total intensity remains the same
    return kernel / np.sum(kernel)


# Example: High stretch along the qx axis
smearing_kernel_PSF_used = generate_elliptical_kernel(size=51, sigma_x=1.0, sigma_y=2.0)

# let's try to build a single intenisty map with the defult values
qx, qy, I = Scatter2D_spherical(rg=5,
                      peak_photon_density=10**11,
                      pixel_count_along_detector=1000,
                      distance_to_detector=150,
                      wavelength=0.15,
                      detector_length=7.0,
                      smearing_kernel_PSF=smearing_kernel_PSF_used,
                      amount_of_radii_fractions=11,
                      radii_fraction_difference=0.08,
                      distribution_function=example_boltz)

plt.figure()
plt.pcolormesh(qx, qy, I, shading='auto')
plt.xlabel("qx")
plt.ylabel("qy")
plt.colorbar(label="I")
plt.show()
