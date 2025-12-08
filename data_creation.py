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
        smearing_kernel_PSF,
        amount_of_radii_fractions,
        radii_fraction_difference,
        distribution_function
):
    """
    Validate and sanitize all input parameters. Return potentially updated values.
    If a parameter is invalid, reset to default and print info.

    Returns
    -------
    Tuple of all parameters, potentially corrected
    """

    # --- Default values from wrapper ---
    default_values = {
        "rg": 1.0,
        "phi_tag_tag": -1 / 63,
        "photon_noise_count": 0,
        "pixel_count_along_detector": 1000,
        "distance_to_detector": 150,
        "wavelength": 0.15,
        "detector_length": 7.0,
        "smearing_kernel_PSF": np.array([[1.0]]),
        "amount_of_radii_fractions": 11,
        "radii_fraction_difference": 0.08,
        "distribution_function": "normalised_gaussian"
    }

    # --- Begin checks ---
    if not isinstance(rg, (int, float)) or rg <= 0:
        print(f"rg={rg} invalid, resetting to default {default_values['rg']}")
        rg = default_values["rg"]

    if not isinstance(phi_tag_tag, (int, float)):
        print(f"phi_tag_tag={phi_tag_tag} invalid, resetting to default {default_values['phi_tag_tag']}")
        phi_tag_tag = default_values["phi_tag_tag"]

    if not isinstance(photon_noise_count, (int, float)) or photon_noise_count < 0:
        print(
            f"photon_noise_count={photon_noise_count} invalid, resetting to default {default_values['photon_noise_count']}")
        photon_noise_count = default_values["photon_noise_count"]

    if not isinstance(pixel_count_along_detector, int) or pixel_count_along_detector < 1:
        print(
            f"pixel_count_along_detector={pixel_count_along_detector} invalid, resetting to default {default_values['pixel_count_along_detector']}")
        pixel_count_along_detector = default_values["pixel_count_along_detector"]

    if not isinstance(distance_to_detector, (int, float)) or distance_to_detector <= 0:
        print(
            f"distance_to_detector={distance_to_detector} invalid, resetting to default {default_values['distance_to_detector']}")
        distance_to_detector = default_values["distance_to_detector"]

    if not isinstance(wavelength, (int, float)) or wavelength <= 0:
        print(f"wavelength={wavelength} invalid, resetting to default {default_values['wavelength']}")
        wavelength = default_values["wavelength"]

    if not isinstance(detector_length, (int, float)) or detector_length <= 0:
        print(f"detector_length={detector_length} invalid, resetting to default {default_values['detector_length']}")
        detector_length = default_values["detector_length"]

    if smearing_kernel_PSF is None or not isinstance(smearing_kernel_PSF, np.ndarray) or smearing_kernel_PSF.ndim != 2:
        print(f"smearing_kernel_PSF invalid, resetting to default {default_values['smearing_kernel_PSF']}")
        smearing_kernel_PSF = default_values["smearing_kernel_PSF"]

    # Check number of discrete radii points: should be odd and >=3
    if not isinstance(amount_of_radii_fractions,
                      int) or amount_of_radii_fractions < 3 or amount_of_radii_fractions % 2 == 0:
        print(
            f"amount_of_radii_fractions={amount_of_radii_fractions} invalid, resetting to default {default_values['amount_of_radii_fractions']}")
        amount_of_radii_fractions = default_values["amount_of_radii_fractions"]

    if not isinstance(radii_fraction_difference, (int, float)) or radii_fraction_difference <= 0:
        print(
            f"radii_fraction_difference={radii_fraction_difference} invalid, resetting to default {default_values['radii_fraction_difference']}")
        radii_fraction_difference = default_values["radii_fraction_difference"]

    if distribution_function is None or (
            not callable(distribution_function) and distribution_function != "normalised_gaussian"):
        print(f"distribution_function={distribution_function} invalid, resetting to default 'normalised_gaussian'")
        distribution_function = default_values["distribution_function"]

    # --- Return validated / corrected parameters ---
    return (
        rg,
        phi_tag_tag,
        photon_noise_count,
        pixel_count_along_detector,
        distance_to_detector,
        wavelength,
        detector_length,
        smearing_kernel_PSF,
        amount_of_radii_fractions,
        radii_fraction_difference,
        distribution_function
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

def smear_intensity(intensity_map, smearing_kernel_PSF):
    """
    Convolve the intensity with the detector PSF kernel.

    smearing_kernel_PSF : 2D np.ndarray
        Kernel for detector smearing (e.g., Gaussian blur)
    """
    return convolve2d(intensity_map, smearing_kernel_PSF, mode="same")


# ============================================================
# ==================== 7. ADD NOISE ==========================
# ============================================================

def add_noise(intensity_map, photon_noise_count):
    """
    Apply Gaussian photon-count noise.

    Parameters
    ----------
    intensity_map : np.ndarray
        NxN intensity grid
    photon_noise_count : float
        Standard deviation of added Gaussian noise
    """
    return intensity_map + photon_noise_count * np.random.randn(*intensity_map.shape)


# ============================================================
# ======================= MAIN WRAPPER ======================
# ============================================================

def Scatter2D(
        rg,
        phi_tag_tag=-1 / 63,
        photon_noise_count=0,
        pixel_count_along_detector=1000,
        distance_to_detector=150,
        wavelength=0.15,
        detector_length=7.0,
        smearing_kernel_PSF=None,
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
    """

    # 1. Validate and sanitize all inputs
    rg, phi_tag_tag, photon_noise_count, pixel_count_along_detector, \
        distance_to_detector, wavelength, detector_length, smearing_kernel_PSF, \
        amount_of_radii_fractions, radii_fraction_difference, distribution_function = validate_scatter_parameters(
            rg, phi_tag_tag, photon_noise_count,
            pixel_count_along_detector, distance_to_detector,
            wavelength, detector_length, smearing_kernel_PSF,
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
    F = smear_intensity(F, smearing_kernel_PSF)

    # 6. Add noise
    F = add_noise(F, photon_noise_count)

    return qx, qy, F
