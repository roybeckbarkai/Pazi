from data_creation import Scatter2D, normalised_gaussian
import matplotlib.pyplot as plt
from data_analysis import get_V_and_phi_tagtag
from distribution_functions import Boltzmann_dis

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
                      smearing_kernel_PSF=[[1, 0], [0, 1]],
                      amount_of_radii_fractions=11,
                      radii_fraction_difference=0.08,
                      distribution_function=example_boltz)

plt.figure()
plt.pcolormesh(qx, qy, I, shading='auto')
plt.xlabel("qx")
plt.ylabel("qy")
plt.colorbar(label="I")
plt.show()

results_list = get_V_and_phi_tagtag(I, qx, qy, q_min=0, q_max=0.01, sigma=1)

# Correctly unpack the tuple found at the 0-index of the list
r_g, V, phi_tag_tag = results_list[0]

print(f"r_g={r_g}, V={V}, phi_tag_tag={phi_tag_tag}")
