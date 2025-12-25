# the same data creation process, but instead of phi_tag_tag we use the proper shape factor
# of a spherical particle
import data_creation as data
import numpy as np

from distribution_functions import gaussian
def normalised_gaussian(x):
    y = gaussian(x, mean=0, std=1)
    return y


def form_factor_intensity_spherical(qx, qy, radii, p_weights, rg):
    """
    Computes the 2D intensity map for spherical particles.
    """
    # Calculate q magnitude
    q = np.sqrt(qx ** 2 + qy ** 2)

    # Initialize intensity map
    total_intensity = np.zeros_like(q)

    # Loop through the distribution of radii
    for r_fraction, weight in zip(radii, p_weights):
        # Convert Radius of Gyration fraction to actual sphere radius
        # R = Rg * sqrt(5/3)
        R = (rg * r_fraction) * np.sqrt(5 / 3)

        # Avoid division by zero at q=0
        qR = q * R
        with np.errstate(divide='ignore', invalid='ignore'):
            # Calculate the general form
            bes_val = 3 * (np.sin(qR) - qR * np.cos(qR)) / (qR ** 3)

            # Patch the center (q=0) where the limit is 1.0
            bes = np.where(qR < 1e-9, 1.0, bes_val)

        total_intensity += weight * (bes ** 2)

    return total_intensity

# repelcate the process for the sphere
def Scatter2D_spherical(
        rg,
        peak_photon_density=0,
        pixel_count_along_detector=1000,
        distance_to_detector=150,
        wavelength=0.15,
        detector_length=7.0,
        smear_dims=(1,1),
        amount_of_radii_fractions=11,
        radii_fraction_difference=0.08,
        distribution_function=normalised_gaussian
):
    """
    High-level 2D scattering simulation wrapper.
    Steps:
    1. Validate parameters
    2. Build q-grid
    3. Compute discrete radius distribution
    4. Build intensity map from physics model
    5. Apply smearing
    6. Add noise

        rg - a float - the average radius of gyration in the sample. In units of nm
        phi_tag_tag - a float - the factor for extended Guinier analysis
        peak_photon_density - a float - the photon density at the peak of the ray - In units of photon/nm**2
        pixel_count_along_detector - a float - the amount of pixels along a vertex of a square detector
        distance_to_detector - a float - distance from the sample to the detector - In units of cm
        wavelength- a float - wavelength of the scattering X-ray - In units of nm
        detector_length - a float - the length of the detector - In units of cm
        smearing_kernel_PSF - a 2D array - kernel used for stretching the intensity map
        amount_of_radii_fractions - a float - number of radii we sample from our pdf
        radii_fraction_difference - a float - relative section of radii we check from Rg
        distribution_function - a function - the distribution function from which we sample radii wights

    """

    # 1. Validate and sanitize all inputs
    # Note: we pass 0 for phi_tag_tag since spheres don't use it
    rg, _, photon_noise_count, pixel_count_along_detector, \
        distance_to_detector, wavelength, detector_length, smear_dims, \
        amount_of_radii_fractions, radii_fraction_difference, distribution_function = data.validate_scatter_parameters(
        rg, 0, peak_photon_density,
        pixel_count_along_detector, distance_to_detector,
        wavelength, detector_length, smear_dims,
        amount_of_radii_fractions, radii_fraction_difference,
        distribution_function
    )

    # 2. Build q-grid
    qx, qy = data.create_q_grid(pixel_count_along_detector, detector_length, distance_to_detector, wavelength)

    # 3. Build discrete radius distribution
    radii, p_weights = data.radius_distribution_model(distribution_function, amount_of_radii_fractions,
                                                 radii_fraction_difference)

    # 4. Build intensity map
    F = form_factor_intensity_spherical(qx, qy, radii, p_weights, rg)

    # 5. Apply smearing
    F_smeared = data.smear_intensity(F, smear_dims)

    # 6. Add noise
    F_full = data.add_noise(F_smeared, peak_photon_density)

    return qx, qy, F_full, F_smeared, F