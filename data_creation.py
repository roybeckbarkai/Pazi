import numpy as np
from scipy.signal import convolve2d
from distribution_functions import gaussian


# ============================================================
# ================ 1. WARNINGS & VALIDATION =================
# ============================================================

def validate_scatter_parameters(
    rg,
    phi_tag_tag,
    photon_noise_count,
    pixel_count_along_detector,
    distance_to_detector,
    wavelength,
    detector_length,
    smear_dims,
    amount_of_radii_fractions,
    radii_fraction_difference,
    distribution_function
):
    """
    Validate input parameters.
    Raises ValueError if any parameter is invalid.
    """

    if not isinstance(rg, (int, float)) or rg <= 0:
        raise ValueError(f"Invalid rg: {rg}. Must be a positive number.")

    if not isinstance(phi_tag_tag, (int, float)):
        raise ValueError(f"Invalid phi_tag_tag: {phi_tag_tag}. Must be numeric.")

    if not isinstance(photon_noise_count, (int, float)) or photon_noise_count < 0:
        raise ValueError(f"Invalid photon_noise_count: {photon_noise_count}. Cannot be negative.")

    if not isinstance(pixel_count_along_detector, int) or pixel_count_along_detector < 1:
        raise ValueError(f"Invalid pixel_count_along_detector: {pixel_count_along_detector}. Must be int >= 1.")

    if not isinstance(distance_to_detector, (int, float)) or distance_to_detector <= 0:
        raise ValueError(f"Invalid distance_to_detector: {distance_to_detector}.")

    if not isinstance(wavelength, (int, float)) or wavelength <= 0:
        raise ValueError(f"Invalid wavelength: {wavelength}.")

    if not isinstance(detector_length, (int, float)) or detector_length <= 0:
        raise ValueError(f"Invalid detector_length: {detector_length}.")

    if (not isinstance(smear_dims, (tuple, list)) or
            len(smear_dims) != 2 or
            not all(isinstance(x, int) and x > 0 for x in smear_dims)):
        raise ValueError(f"smear_dims must be a tuple of two positive integers (m, n). Got: {smear_dims}")

    if not isinstance(amount_of_radii_fractions, int) or amount_of_radii_fractions < 3 or amount_of_radii_fractions % 2 == 0:
        raise ValueError(f"amount_of_radii_fractions ({amount_of_radii_fractions}) must be an odd integer >= 3.")

    if not isinstance(radii_fraction_difference, (int, float)) or radii_fraction_difference <= 0:
        raise ValueError(f"Invalid radii_fraction_difference: {radii_fraction_difference}.")

    if distribution_function is None or (not callable(distribution_function) and distribution_function != "normalised_gaussian"):
        raise ValueError(f"Invalid distribution_function: {distribution_function}.")

    return (
        rg, phi_tag_tag, photon_noise_count, pixel_count_along_detector,
        distance_to_detector, wavelength, detector_length, smear_dims,
        amount_of_radii_fractions, radii_fraction_difference, distribution_function
    )


# ============================================================
# =================== 2. CREATE q-GRID ========================
# ============================================================

def create_q_grid(
        pixel_count_along_detector,
        detector_length,
        distance_to_detector,
        wavelength
):
    """
    Create qx, qy grids for a square detector.

    Parameters
    ----------
    pixel_count_along_detector : int
        Number of pixels along one dimension of the detector (NxN).
    detector_length : float
        Physical size of the detector (cm).
    distance_to_detector : float
        Sample-to-detector distance (cm).
    wavelength : float
        X-ray wavelength (nm).

    Returns
    -------
    qx, qy : 2D numpy arrays
        qx and qy grids of shape (N, N).

    Notes
    -----
    This implements a simple square detector mapping. Each pixel is centered
    and the origin is at the center of the detector. Conversion from real-space
    (cm) to q-space follows q = (4*pi / wavelength) * (x / (2*distance)).
    """
    pixel_length = detector_length / pixel_count_along_detector

    # real-space detector coordinates (cm)
    # 1. Create pixel indices: 0, 1, 2, ..., N-1
    indices = np.arange(pixel_count_along_detector)

    # 2. Compute the center index (works for even or odd N)
    center_index = (pixel_count_along_detector - 1) / 2

    # 3. Shift indices so the detector is centered around zero
    centered_indices = indices - center_index

    # 4. Convert index units → physical length units (cm)
    x = centered_indices * pixel_length

    # move from 1D to 2D
    X, Y = np.meshgrid(x, x)

    # conversion from detector coordinate to q
    conversion = (4 * np.pi / wavelength) * (1 / (2 * distance_to_detector))

    qx = conversion * X
    qy = conversion * Y

    return qx, qy


# ============================================================
# =================== 3. GENERAL FORM FACTOR =================
# ============================================================

def general_form_factor(value_squared, phi_tag_tag):
    """
    Monolithic form factor f(q) = exp(-q^2/3 + 0.5 * φ'' * q^4)

    Parameters
    ----------
    value_squared : np.ndarray
        Precomputed q^2 * rg^2 values for the current radius.
    phi_tag_tag : float
        The fourth-order coefficient in the expansion.

    Returns
    -------
    ff : np.ndarray
        Form factor evaluated at each point.
    """
    ff = np.exp(-value_squared / 3 + 0.5 * phi_tag_tag * value_squared ** 2)
    return ff


# ============================================================
# ================= 4. RADIUS DISTRIBUTION ===================
# ============================================================

def radius_distribution_model(distribution_function, amount_of_radii_fractions, radii_fraction_difference):
    """
    Build the discrete set of radii fractions and compute weights.

    Parameters
    ----------
    distribution_function : callable
        External function to compute probability for a given radius fraction.
    amount_of_radii_fractions : int
        Number of discrete radii fractions to consider (should be odd).
    radii_fraction_difference : float
        Step between each discrete radius fraction.

    Returns
    -------
    r_values : np.ndarray
        Fractional deviations from rg, e.g., [-0.1, -0.05, 0, 0.05, 0.1].
    p_weights : np.ndarray
        Probability weights from distribution_function.
    """
    # build the discrete fractions
    m = (amount_of_radii_fractions - 1) // 2
    r_values = np.arange(-m, m + 1) * radii_fraction_difference

    # compute probability weights
    if distribution_function == "normalised_gaussian":
        # placeholder, will define normalized Gaussian later
        p_weights = np.exp(-r_values ** 2 / (2 * np.var(r_values)))
        p_weights /= np.sum(p_weights)
    else:
        p_weights = distribution_function(r_values)

    return r_values, p_weights


# ============================================================
# ===== 4b. SIMPLE DISTRIBUTION OPTION FOR LATER USE==========
# ============================================================

def normalised_gaussian(x):
    y = gaussian(x, mean=0, std=1)
    return y


# ============================================================
# ========== 5. INTENSITY FOR A SINGLE RADIUS VALUE ==========
# ============================================================

def form_factor_intensity(qx, qy, radii, p_weights, rg, phi_tag_tag):
    """
    Compute intensity map F over qx, qy for all radii in radii[].

    Parameters
    ----------
    qx, qy : np.ndarray
        q-space grids.
    radii : np.ndarray
        Fractional radii deviations.
    p_weights : np.ndarray
        Probability weights for each radius fraction.
    rg : float
        Mean radius.
    phi_tag_tag : float
        Fourth-order form factor coefficient.

    Returns
    -------
    F : np.ndarray
        Intensity map of shape qx.shape
    """
    q_squared = qx ** 2 + qy ** 2
    F = np.zeros_like(q_squared, dtype=float)

    for i, r_frac in enumerate(radii):
        rg_mod = rg * (1 + r_frac)
        value_squared = q_squared * rg_mod ** 2
        f = general_form_factor(value_squared, phi_tag_tag)
        F += f * p_weights[i]

    # --- Optional: eliminate anomalies outside Guinier region ---
    # This masks unphysical points where the q^4 term dominates over the Guinier q^2 term
    mask = np.abs(-q_squared / 3) < np.abs(0.5 * phi_tag_tag * q_squared ** 2)
    F[mask] = np.nan

    return F


# ============================================================
# ===================== 6. SMEARING ==========================
# ============================================================

def make_bartlett_kernel(m, n):
    """Creates a normalized 2D Bartlett (triangular) kernel."""
    # np.bartlett(m+2)[1:-1] ensures the edges are non-zero
    wy = np.bartlett(m + 2)[1:-1]
    wx = np.bartlett(n + 2)[1:-1]
    kernel = np.outer(wy, wx)
    return kernel / np.sum(kernel)


def make_kernel_odd_centered(kernel_in):
    """Expands even-sized kernel to odd while keeping center (MATLAB logic)."""
    kernel_out = np.array(kernel_in, dtype=float)

    # Rows expansion
    r, c = kernel_out.shape
    if r % 2 == 0:
        tmp = np.zeros((r + 1, c))
        for i in range(r):
            tmp[i, :] += 0.5 * kernel_out[i, :]
            tmp[i + 1, :] += 0.5 * kernel_out[i, :]
        kernel_out = tmp

    # Columns expansion
    r, c = kernel_out.shape
    if c % 2 == 0:
        tmp = np.zeros((r, c + 1))
        for j in range(c):
            tmp[:, j] += 0.5 * kernel_out[:, j]
            tmp[:, j + 1] += 0.5 * kernel_out[:, j]
        kernel_out = tmp
    return kernel_out


def smear_intensity(intensity_map, smear_dims):
    """
    Creates an odd-centered kernel from (m, n) and smears the intensity map.
    """
    m, n = smear_dims
    raw_kernel = make_bartlett_kernel(m, n)
    centered_kernel = make_kernel_odd_centered(raw_kernel)

    # mode="same" keeps output size identical to input size
    return convolve2d(intensity_map, centered_kernel, mode="same")


# ============================================================
# ==================== 7. ADD NOISE ==========================
# ============================================================

def add_noise(intensity_map, peak_photon_density):
    """
    Apply Gaussian photon-count noise.

    Parameters
    ----------
    intensity_map : np.ndarray
        NxN intensity grid
    peak_photon_density : float
        peak photon density at the center of the bean, in units of photon/nm**2
    """
    return intensity_map + np.sqrt(intensity_map / peak_photon_density) * np.random.randn(*intensity_map.shape)


# ============================================================
# ======================= MAIN WRAPPER ======================
# ============================================================

def Scatter2D(
        rg,
        phi_tag_tag=-1 / 63,
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
    rg, phi_tag_tag, photon_noise_count, pixel_count_along_detector, \
        distance_to_detector, wavelength, detector_length, smear_dims, \
        amount_of_radii_fractions, radii_fraction_difference, distribution_function = validate_scatter_parameters(
            rg, phi_tag_tag, peak_photon_density,
            pixel_count_along_detector, distance_to_detector,
            wavelength, detector_length, smear_dims,
            amount_of_radii_fractions, radii_fraction_difference,
            distribution_function
            )

    # 2. Build q-grid
    qx, qy = create_q_grid(pixel_count_along_detector, detector_length, distance_to_detector, wavelength)

    # 3. Build discrete radius distribution
    radii, p_weights = radius_distribution_model(distribution_function, amount_of_radii_fractions,
                                                 radii_fraction_difference)

    # 4. Build intensity map
    F = form_factor_intensity(qx, qy, radii, p_weights, rg, phi_tag_tag)

    # 5. Apply smearing
    F_smeared = smear_intensity(F, smear_dims)

    # 6. Add noise
    F = add_noise(F_smeared, peak_photon_density)

    return qx, qy, F
