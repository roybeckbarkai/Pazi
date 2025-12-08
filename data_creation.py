import numpy as np
import scipy as sp
from scipy.signal import convolve2d

# our code is aimed at simulating data, meaning (I, qx, qy) for us to analysis robustness of algorithems o
# ============================================================
# ===================== HELPER FUNCTIONS ======================
# ============================================================

def makeKernelOddCentered(kernel_in):
    """
    makeKernelOddCentered:
    Expand even-sized kernel to odd while keeping center.
    - If rows are even → expand to rows+1, splitting row weights 50/50
    - If cols are even → expand to cols+1, splitting column weights 50/50

    MATLAB examples:
        makeKernelOddCentered([1 1])        -> [0.5 1 0.5]
        makeKernelOddCentered([1;1])       -> [0.5; 1; 0.5]
        makeKernelOddCentered([1 2;3 4])   -> 3x3 result
    """
    kernel_out = np.array(kernel_in, dtype=float)
    r, c = kernel_out.shape

    # Expand rows if even
    if r % 2 == 0:
        newr = r + 1
        tmp = np.zeros((newr, c))
        for i in range(r):
            tmp[i, :] += 0.5 * kernel_out[i, :]
            tmp[i + 1, :] += 0.5 * kernel_out[i, :]
        kernel_out = tmp

    r, c = kernel_out.shape

    # Expand columns if even
    if c % 2 == 0:
        newc = c + 1
        tmp = np.zeros((r, newc))
        for j in range(c):
            tmp[:, j] += 0.5 * kernel_out[:, j]
            tmp[:, j + 1] += 0.5 * kernel_out[:, j]
        kernel_out = tmp

    return kernel_out


def gaussian_discrete(N, V, alpha=0.5):
    """
    gaussian_discrete:
    Discrete symmetric Gaussian with exact variance V.

    MATLAB doc:
        Returns:
          x : Nx1 locations (symmetric)
          p : Nx1 probs (sum=1)
        Such that:
          sum((x - μ)^2 * p) = V, with μ=0 by symmetry
    """
    if V == 0:
        x = np.zeros(N)
        p = np.ones(N) / N
        return x, p

    # symmetric discrete grid
    if N % 2 == 1:
        m = (N - 1) // 2
        u = np.arange(-m, m + 1)
    else:
        m = N // 2
        half = np.arange(m) + 0.5
        u = np.concatenate([-half[::-1], half])

    # Gaussian weights
    w = np.exp(-alpha * u**2)

    # enforce symmetry
    for i in range(N // 2):
        avg = 0.5 * (w[i] + w[-i - 1])
        w[i] = w[-i - 1] = avg

    p = w / np.sum(w)

    # scale so that variance = V exactly
    Eu2 = np.sum((u**2) * p)
    a = np.sqrt(V / Eu2)

    x = a * u
    return x, p


def tri1d(N, mode="raised"):
    """1D triangular window (MATLAB: Bartlett)."""
    if N <= 0:
        return np.zeros(0)
    if N == 1:
        return np.array([1.0])

    idx = np.arange(N)
    c = (N - 1) / 2

    if mode == "zero":
        if N == 2:
            w = np.array([1.0, 1.0])
        else:
            w = 1 - np.abs(idx - c) / c
    else:  # raised
        d = (N + 1) / 2
        w = 1 - np.abs(idx - c) / d
        w[w < 0] = 0

    return w


def bartlett2d(n, m=None, mode="raised"):
    """
    bartlett2d:
    Normalized 2D triangular window.
    MATLAB equivalent: uses tri1d separable kernel.
    """
    if m is None:
        m = n

    wy = tri1d(n, mode).reshape(-1, 1)
    wx = tri1d(m, mode).reshape(1, -1)
    K = wy @ wx

    s = np.sum(K)
    return K / s if s > 0 else K


# ============================================================
# ====================== MAIN FUNCTION ========================
# ============================================================

def Scatter2D(
    rg,                             # mean radius
    Nois,                           # noise level
    V,                              # variance of radii distribution
    Nu=-1/63,                       # scattering parameter (default spherical)
    DETpix=1000,                    # number of detector pixels
    SD_dist=150,                    # sample-detector distance (cm)
    lambda_=0.15,                   # wavelength (nm)
    det_side=7.0,                   # detector side length (cm)
    PSF0=None                       # point-spread function kernel
):
    """
    Scatter2D:
    Python translation of MATLAB function Scatter2D.

    ----------------------------------------------------------
    MATLAB notes:

    simulate a specific scattering case with noise added
    rg, Nu, V are the scattering parameters.
    Nois = noise level. Negative Nois means photon count model.
    Detpix = number of pixels.
    SD_dist = sample-detector distance (cm)
    det_side = full detector side length (cm)
    lambda = wavelength (nm)
    PSF0 = point-spread function (pre-convolution)

    Returns:
        q_mat_x, q_mat_y : meshgrid of q-values
        I_mat            : intensity matrix

    Outside Guinier region → intensity = NaN
    ----------------------------------------------------------
    """

    # --- Ensure PSF exists ---
    if PSF0 is None:
        PSF0 = np.ones((1, 1))

    PSF0 = PSF0 / np.sum(PSF0)
    PSF0 = makeKernelOddCentered(PSF0)

    # --- Clean main parameters ---
    rg = abs(rg)
    V = abs(V)

    # single assignment:
    r_g = rg

    # --- Compute q-grid ---
    det_hside = det_side / 2
    maxq = 4 * np.pi / lambda_ * det_hside / SD_dist

    Npix = int(round(DETpix / 2))
    qv = np.linspace(-maxq, maxq, 2 * Npix + 1)

    qvx, qvy = np.meshgrid(qv, qv)  # MATLAB: meshgrid(qv)
    qvr2 = qvx**2 + qvy**2
    qvr4 = qvr2 * qvr2
    qvr = np.sqrt(qvr2)

    # --- Monolithic form factor ---
    ff = lambda qr: np.exp(-qr**2 / 3 + 0.5 * Nu * qr**4)

    # --- Radius distribution ---
    n_radii = 11
    r_vect, p = gaussian_discrete(n_radii, V)

    F = np.zeros_like(qvr2)

    # --- Scattering intensity ---
    for ii in range(n_radii):
        s = r_g * (1 + r_vect[ii])
        s2 = s * s
        s4 = s2 * s2

        exponent = -(s2/3) * qvr2 + (0.5 * Nu * s4) * qvr4

        # eliminate anomalies outside guinier region
        mask = np.abs(-(s2/3) * qvr2) < np.abs((0.5 * Nu * s4) * qvr4)
        exponent[mask] = np.nan

        F += p[ii] * np.exp(exponent)

    # --- Apply detector PSF ---
    F = convolve2d(F, PSF0, mode="same")

    # --- Noise model ---
    if Nois < 0:   # photon noise
        F *= abs(Nois)                       # scale
        F = F + np.sqrt(np.maximum(F, 0)) * np.random.randn(*F.shape)
        F /= abs(Nois)
    else:          # Gaussian noise
        F = F + Nois * np.random.randn(*F.shape)

    return qvx, qvy, F


"""NEW FORM"""

import numpy as np
from scipy.signal import convolve2d


# ============================================================
# ================ 1. WARNINGS & VALIDATION  ==================
# ============================================================

def validate_scatter_parameters(
        rg,
        variance,
        phi_tag_tag,
        photon_noise_count,
        pixel_amount_detector,
        distance_to_detector,
        wavelength,
        detector_size,
        psf_kernel
):
    """
    Validate and sanitize all input parameters.
    (Range checks, type checks, physical constraints, etc.)
    """
    pass


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
    """

    pixel_length = detector_length / pixel_count_along_detector

    # real-space detector coordinates (cm)
    # 1. Create pixel indices: 0, 1, 2, ..., N-1
    indices = np.arange(pixel_count_along_detector)

    # 2. Compute the center index (works for even or odd N)
    center_index = (pixel_count_along_detector - 1) / 2

    # 3. Shift indices so the detector is centered around zero
    #    e.g. [-2, -1, 0, 1, 2] for N=5
    centered_indices = indices - center_index

    # 4. Convert index units → physical length units (cm)
    #    Each pixel corresponds to a physical width pixel_length
    x = centered_indices * pixel_length

    # 5. Move from 1D → 2D
    X, Y = np.meshgrid(x, x)

    # 6. Conversion from detector coordinates → q
    #    Small-angle approx: q = (4π/λ) * (r / 2D)
    conversion = (4 * np.pi / wavelength) * (1 / (2 * distance_to_detector))
    qx = conversion * X
    qy = conversion * Y

    return qx, qy


# ============================================================
# =================== 3. GENERAL FORM FACTOR =================
# ============================================================

def general_form_factor(value, phi_tag_tag):
    """
    General monolithic form factor:
    f(value) = exp(-value^2 / 3 + 0.5 * φ'' * value^4)

    Parameters
    ----------
    value : float or array-like
        The variable over which the form factor is evaluated (q, q^2, etc.).
    phi_tag_tag : float
        Scattering parameter controlling higher-order term.

    Returns
    -------
    ff : float or array-like
        Form factor evaluated at the input value.
    """
    ff = np.exp(-value ** 2 / 3 + 0.5 * phi_tag_tag * value ** 4)
    return ff


# ============================================================
# =================== 4. RADIUS DISTRIBUTION ==================
# ============================================================

def radius_distribution_model(distribution_function, variance):
    """
    Return (r_values, probabilities) for the chosen
    distribution function describing particle radii.
    """
    pass


# ============================================================
# ========= 5. INTENSITY FOR A SINGLE RADIUS VALUE ============
# ============================================================

def intensity_single_radius(qr2, qr4, rg, r_i, phi_tag_tag):
    """
    Compute the intensity contribution from a single radius r_i.
    """
    pass


# ============================================================
# ===================== 6. SMEARING ===========================
# ============================================================

def smear_intensity(intensity_map, psf_kernel):
    """
    Convolve the intensity with the detector PSF kernel.
    """
    pass


# ============================================================
# ==================== 7. ADD NOISE ===========================
# ============================================================

def add_noise(intensity_map, photon_noise_count):
    """
    Apply Gaussian or photon-count noise to a simulated intensity.
    """
    pass


# ============================================================
# ======================= MAIN WRAPPER ========================
# ============================================================

def Scatter2D(
        rg,
        variance,
        phi_tag_tag=-1 / 63,
        photon_noise_count=0,
        pixel_amount_detector=1000,
        distance_to_detector=150,
        wavelength=0.15,
        detector_size=7.0,
        psf_kernel=None,
        distribution_function=None
):
    """
    High-level interface for simulating 2D scattering.
    """
    # 1. Validate
    validate_scatter_parameters(
        rg, variance, phi_tag_tag, photon_noise_count,
        pixel_amount_detector, distance_to_detector,
        wavelength, detector_size, psf_kernel
    )

    # 2. q-grid
    qx, qy = create_q_grid(
        pixel_amount_detector,
        detector_size,
        distance_to_detector,
        wavelength
    )

    # 3. radius distribution
    r_values, p_weights = radius_distribution_model(
        distribution_function,
        variance
    )

    # 4. Build intensity map
    I = np.zeros_like(qx)

    for r_i, p_i in zip(r_values, p_weights):
        I += p_i * intensity_single_radius(qx ** 2 + qy ** 2, (qx ** 2 + qy ** 2) ** 2, rg, r_i, phi_tag_tag)

    # 5. smear
    I = smear_intensity(I, psf_kernel)

    # 6. add noise
    I = add_noise(I, photon_noise_count)

    return qx, qy, I