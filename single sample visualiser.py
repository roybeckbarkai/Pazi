import matplotlib.pyplot as plt
import numpy as np

from distribution_functions import Boltzmann_dis
from data_creation import Scatter2D

def example_boltz(x):
    y = Boltzmann_dis(x, sigma=0.1, mean=1)
    return y


# let's try to build a single intenisty map with the defult values
qx, qy, I = Scatter2D(rg=1,
                      phi_tag_tag=-1 / 63,
                      peak_photon_density=10**11,
                      pixel_count_along_detector=1000,
                      distance_to_detector=150,
                      wavelength=0.15,
                      detector_length=7.0,
                      smear_dims=(1,51),
                      amount_of_radii_fractions=11,
                      radii_fraction_difference=0.08,
                      distribution_function=example_boltz)

plt.figure()
plt.pcolormesh(qx, qy, I, shading='auto')
plt.xlabel("qx[1/nm]")
plt.ylabel("qy[1/nm]")
plt.colorbar(label="I")
plt.show()

