import matplotlib.pyplot as plt
import numpy as np

from distribution_functions import Boltzmann_dis
from data_creation_spherical import Scatter2D_spherical

def example_boltz(x):
    y = Boltzmann_dis(x, sigma=0.1, mean=1)
    return y

# let's try to build a single intenisty map with the defult values
qx, qy, _,F_smeared, F_pre_smear = Scatter2D_spherical(rg=5,
                      peak_photon_density=10**11,
                      pixel_count_along_detector=1000,
                      distance_to_detector=150,
                      wavelength=0.15,
                      detector_length=7.0,
                      smear_dims=(1,3),
                      amount_of_radii_fractions=11,
                      radii_fraction_difference=0.08,
                      distribution_function=example_boltz)



# Plot 3
# 1. Create a boolean mask for the region of interest
# This finds indices where BOTH qx and qy are within [-0.75, 0.75]
sub_area_mask = (np.abs(qx) <= 0.9) & (np.abs(qy) <= 0.9)

# 2. Slice the arrays
# We use the mask to find the bounding box of the data
# This assumes qx and qy are structured grids (standard for pcolormesh)
rows = np.any(sub_area_mask, axis=1)
cols = np.any(sub_area_mask, axis=0)

qx_sub = qx[rows][:, cols]
qy_sub = qy[rows][:, cols]
diff_sub = F_smeared [rows][:, cols]- F_pre_smear[rows][:, cols]

# 3. Plot the sub-area
plt.figure(figsize=(6, 5))
cmap = plt.get_cmap('jet', 20)
plt.pcolormesh(qx_sub, qy_sub, diff_sub, shading='auto', cmap=cmap)
plt.title("Difference: Sub-area [-0.75, 0.75]")
plt.colorbar(label="Intensity Difference")
plt.show()