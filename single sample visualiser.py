from data_creation import Scatter2D, normalised_gaussian
import matplotlib.pyplot as plt

# let's try to build a single intenisty map with the defult values
qx, qy, I = Scatter2D(rg=1,
                      phi_tag_tag=-1 / 63,
                      photon_noise_count=0,
                      pixel_count_along_detector=1000,
                      distance_to_detector=150,
                      wavelength=0.15,
                      detector_length=7.0,
                      smearing_kernel_PSF=[[1, 0], [0, 1]],
                      amount_of_radii_fractions=11,
                      radii_fraction_difference=0.08,
                      distribution_function=normalised_gaussian)

plt.figure()
plt.pcolormesh(qx, qy, I, shading='auto')
plt.xlabel("qx")
plt.ylabel("qy")
plt.colorbar(label="I")
plt.show()
