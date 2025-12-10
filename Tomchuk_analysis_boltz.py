import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erf

from data_creation import Scatter2D
from distribution_functions import Boltzmann_dis

# first - define extended guiner
# --- Step 1: Define the Guinier Fit Function ---

def guinier_fit_ext(q_squared, G, R_g, B):
    """
    The full Guinier/Porod combined fit function.

    Parameters:
    - q_squared (array): The independent variable, q^2.
    - G (float): Guinier prefactor (I(0) or G).
    - R_g (float): Radius of Gyration.
    - B (float): Porod prefactor.

    Note: The input to the function is q^2, but the equation uses q.
          We derive q from the input q^2.
    """
    q = np.sqrt(q_squared)

    # Calculate the two main terms
    # 1. Guinier Term: G * exp(-q^2 * R_g^2 / 3)
    guinier_term = G * np.exp(-q_squared * R_g ** 2 / 3.0)

    # 2. Porod Term (Including the smoothing/transition function)
    #    B*q^-4 * [erf(q*R_g / sqrt(6))]^12

    # To avoid division by zero when q is 0, we use np.where
    # and assume the Porod term is 0 where q=0, or handle q>0 separately

    # Calculate the arguments
    q_pow_minus_4 = np.zeros_like(q)
    # Set q^-4 only where q is not zero
    q_pow_minus_4[q > 0] = q[q > 0] ** (-4)

    erf_arg = q * R_g / np.sqrt(6.0)

    porod_term = B * q_pow_minus_4 * np.power(np.fmin(1.0, erf(erf_arg)), 12)

    # Total Intensity
    I_model = guinier_term + porod_term

    return I_model


## --- Step 2 & 3: Averaging and Fitting Function ---

def perform_guinier_analysis_ext(I, qx, qy, num_bins=50):
    """
    Performs the radial averaging and fits the combined Guinier/Porod model.

    Parameters:
    - I (np.ndarray): NxN intensity matrix.
    - qx (np.ndarray): NxN qx matrix.
    - qy (np.ndarray): NxN qy matrix.
    - num_bins (int): Number of bins for radial averaging.

    Returns:
    - tuple: (G, R_g, B, perr) - The three fit parameters and the standard
             deviation errors of the parameters.
    - tuple: (q_squared_avg, I_avg) - The averaged 1D data used for fitting.
    """

    # Calculate q^2 = qx^2 + qy^2
    q_squared_2D = qx ** 2 + qy ** 2

    # Flatten the arrays to 1D for binning
    I_1D = I.flatten()
    q_squared_1D = q_squared_2D.flatten()

    # --- Radial Averaging (Binning by q^2) ---
    # Determine the range for binning
    min_q_squared = np.min(q_squared_1D[q_squared_1D > 0])  # Avoid q=0
    max_q_squared = np.max(q_squared_1D)

    # Create bins
    bins = np.linspace(min_q_squared, max_q_squared, num_bins + 1)

    # Use np.digitize to assign each q^2 value to a bin index
    bin_indices = np.digitize(q_squared_1D, bins)

    # Initialize arrays for binned data
    I_sum = np.zeros(num_bins)
    q_squared_sum = np.zeros(num_bins)
    count = np.zeros(num_bins, dtype=int)

    # Loop through the data and accumulate sums in the bins
    for i in range(len(I_1D)):
        bin_idx = bin_indices[i] - 1  # Adjust index to be 0-based
        if 0 <= bin_idx < num_bins:
            I_sum[bin_idx] += I_1D[i]
            q_squared_sum[bin_idx] += q_squared_1D[i]
            count[bin_idx] += 1

    # Calculate the averages, filtering out empty bins
    valid_bins = count > 0
    I_avg = I_sum[valid_bins] / count[valid_bins]
    q_squared_avg = q_squared_sum[valid_bins] / count[valid_bins]

    # --- Fitting the 1D Data ---

    # Initial Guess for the parameters (critical for non-linear fits)
    # A simple guess:
    # G ~ Maximum of I_avg (I(0))
    # R_g ~ A reasonable estimate, e.g., based on q_min/q_max range
    # B ~ A small positive value
    G_guess = I_avg[0] if len(I_avg) > 0 else 1.0
    R_g_guess = np.sqrt(3.0 / q_squared_avg[1]) if len(q_squared_avg) > 1 else 1.0  # A heuristic guess
    B_guess = G_guess * 0.01  # Small value

    p0 = [G_guess, R_g_guess, B_guess]

    # Perform the non-linear fit
    try:
        # p_opt: Optimized parameters (G, R_g, B)
        # p_cov: Covariance matrix
        p_opt, p_cov = curve_fit(
            f=guinier_fit_ext,
            xdata=q_squared_avg,
            ydata=I_avg,
            p0=p0,
            # Bounds can often help guide the fit, e.g., R_g and B must be > 0
            bounds=([0, 0, 0], [np.inf, np.inf, np.inf])
        )

        # Calculate the standard deviation (standard error) of the parameters
        # Square root of the diagonal elements of the covariance matrix
        perr = np.sqrt(np.diag(p_cov))

        G, R_g, B = p_opt

        return (G, R_g, B, perr), (q_squared_avg, I_avg)

    except RuntimeError as e:
        print(f"Fit failed: {e}")
        # Return None or appropriate failure signal
        return None, (q_squared_avg, I_avg)


# then calculate PDI
def PDI_calc(G, Rg, B):
    PDI = (50 / 81) * ((B * (Rg ** 4)) / G)
    return PDI


def get_PDI(I, qx, qy, num_bins=50):
    fit_results, averaged_data = perform_guinier_analysis_ext(I, qx, qy, num_bins)
    G_fit, R_g_fit, B_fit, perr = fit_results
    PDI = PDI_calc(G_fit, R_g_fit, B_fit)
    return PDI


# plug into p formula
def p_calc(x):
    coeffs = [2.103 * 10 ** -6, -8.236 * 10 ** -5, 0.001323, 0.01087, 0.0429, -0.007363, -0.6952, 3.146, -6.594, 7.004,
              -2.863]
    p = np.poly1d(coeffs)
    return p(x)


def get_p(I, qx, qy, num_bins=50):
    PDI = get_PDI(I, qx, qy, num_bins=50)
    p = p_calc(PDI)
    return p

# check example


def example_boltz(x):
    y = Boltzmann_dis(x, sigma=0.1, mean=1)
    return y

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
                      distribution_function=example_boltz)

p = get_p(I, qx, qy)
print(p)