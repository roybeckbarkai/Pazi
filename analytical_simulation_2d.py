from statistics import variance
import numpy as np
import pandas
from numba.core.cgutils import sizeof
import form_factor_methods
from scipy.ndimage import gaussian_filter
from typing import Tuple, Optional, Callable, Union, List

# Creates q 2d array based on simulation parameters
def create_q_table(px_number, px_size, sample_detector_distance, wavelength, beam_center: Optional[float]=[0,0]):
    # Generate xy_table and calculate qtable (these values do not change across iterations)
    # If beam center not defined
    beam_center = [px_number[0] / 2, px_number[1] / 2]  # px
        #detector_center = [np.round(px_size[0] / 2), np.round(px_size[1] / 2)]  # px
        #CORRECT THE DETECTOR_CENTER DETERMINATION
    xy_table = np.indices((px_number[0], px_number[1]), dtype=float)
    r = px_size * np.sqrt((beam_center[0] - xy_table[0]) ** 2 + (beam_center[1] - xy_table[1]) ** 2)
    sint = np.sin(0.5 * np.arctan(r / sample_detector_distance))
    qtable = 4 * np.pi * sint / wavelength
    return qtable

# Calculates 2d count array
def analytical_calculate_single_2d (form_factor_name, rg, variance, sigma_x, sigma_y, q_table, px_number, px_size, sample_detector_distance, wavelength):
    # Initialize the detector array based in pixel number
    detector_array = np.zeros((px_number[0], px_number[1])) #MAKE SIZE_LIKE Q_TABLE

    # Generate the distribution of Rgs
    rg_array, distribution_ = form_factor_methods.generate_gaussian_distribution(rg, np.sqrt(variance) * rg)

    # Get form factor function name for call from form_factor_methods
    ff_function = getattr(form_factor_methods, form_factor_name, None)
    if not callable(ff_function):
        raise ValueError(f"Function '{form_factor_name}' not found")

    # For every pixel in the detector, calculate the intensity from the form factor
    for i in range(px_number[0]):
        for j in range(px_number[1]):
            detector_array[i, j] += form_factor_methods.polyP(q_table[i, j], ff_function, rg_array, distribution_)*(px_size ** 2) / (4 * np.pi * sample_detector_distance ** 2) * (1 - ((wavelength * q_table[i,j]) ** 2 / 8 / np.pi ** 2)) ** 3


    # Gaussian convolution of the 2D array based on the given sigmas
    Convoluted_detector_array = gaussian_filter(detector_array, sigma=[sigma_x, sigma_y])

    return Convoluted_detector_array

#Flatten and normalize intensity data from 2D arrays to 1D arrays
'''
    This function takes 2D arrays of radial distances and corresponding intensity counts,
    flattens them, and applies normalization and optional filtering. It can return either
    unique radial values with averaged intensities or the full flattened arrays.

    Parameters:
    -----------
    q_array : 2D array of radial distances.
    count_array : 2D array of intensity counts corresponding to q_array.
    px_size : linear size of a single pixel.
    wavelength : Wavelength of the radiation used.
    distance : Sample-to-detector distance.
    normalization_on: bool
        A flag that allows to choose if the flattening should be done with normalization
            True: with normalization
            False: without normalization
    return_unique : bool, optional
        If True, return unique radial values with averaged intensities (default is True).
    q_min : float, optional
        Minimum q value for filtering (only applied if q_max is also provided).
    q_max : float, optional
        Maximum q value for filtering.
    '''
def flatten_intensity(q_array: np.ndarray,
                      count_array: np.ndarray,
                      px_size: float,
                      wavelength: float,
                      sample_detector_distance: float,
                      normalization_on: Optional[bool] = True,
                      return_unique: Optional[bool] = False,
                      q_min: Optional[float] = None,
                      q_max: Optional[float] = None):

    # Flatten the arrays
    sorted_idx = np.argsort(q_array.flatten())
    q_flattened = q_array.flatten()[sorted_idx]
    intensity_flattened = count_array.flatten()[sorted_idx]
    if return_unique:
        # Create a DataFrame
        df = pandas.DataFrame({'q': q_flattened, 'count': intensity_flattened})
        # Group by the 'q' values and compute the mean of the 'count' values
        grouped_df = df.groupby('q', as_index=False)['count'].mean()
        # Extract the unique R values and their corresponding average counts
        unique_Q = grouped_df['q'].values
        average_counts = grouped_df['count'].values
        # Compute the normalization factor
        norm_factor = unique_Q*4*np.pi * sample_detector_distance ** 2 / (px_size**2) * (1 - ((wavelength*unique_Q) ** 2) / 8 / np.pi**2)** -3
        # Mask the data based on the given q_min and q_max if applicable
        if (q_min is not None) and (q_max is not None):
            mask = (unique_Q >= q_min) & (unique_Q <= q_max)
        elif (q_min is not None):
            mask = (unique_Q >= q_min)
        elif (q_max is not None):
            mask = (unique_Q <= q_max)
        else: mask = True
        #Check if the normalization is applied
        if normalization_on:
            return unique_Q[mask], (average_counts * norm_factor)[mask]
        else:
            return unique_Q[mask], average_counts[mask]
    else:
        if normalization_on:
            # Return the flattened and sorted arrays directly
            norm_factor=4*np.pi * sample_detector_distance ** 2 / (px_size**2) * (1 - ((wavelength*q_flattened) ** 2) / 8 / np.pi**2)** -3
            # Mask the data based on the given q_min and q_max if applicable
            if (q_min is not None) and (q_max is not None):
                mask = (q_flattened >= q_min) & (q_flattened <= q_max)
            elif (q_min is not None):
                mask = (q_flattened >= q_min)
            elif (q_max is not None):
                mask = (q_flattened <= q_max)
            else:
                mask = True
            return q_flattened[mask], (intensity_flattened * norm_factor)[mask]
        else:
            mask = (q_flattened >= q_min) & (q_flattened <= q_max)
            return q_flattened[mask], intensity_flattened[mask]

def single_analytical_simulation_flattened(params):
    """
    Runs a single 2D analytical simulation and returns q_table and I_flattened.

    Parameters:
        params (dict): Dictionary containing simulation parameters.

    Returns:
        tuple: (q_table, I_flattened)
    """

    # Creates q table
    q_table = create_q_table(
        params["px_number"], params["px_size"],
        params["sample_detector_distance"], params["wavelength"]
    )

    # Runs the analytical calculation of 2D intensity profile
    # Outputs 2D intensity distribution: I_array
    I_array = analytical_calculate_single_2d(
        params["form_factor_name"], params["rg"], params["variance"],
        params["sigma_x"], params["sigma_y"], q_table,
        params["px_number"], params["px_size"],
        params["sample_detector_distance"], params["wavelength"]
    )

    # Flattens the data -- creates .dat file with 1D Intensity vs q
    q_flattened, I_flattened = flatten_intensity(
        q_table, I_array,
        px_size=params["px_size"],
        wavelength=params["wavelength"],
        sample_detector_distance=params["sample_detector_distance"],
        normalization_on=params["normalization_on"],
        return_unique=params["return_unique"],
        q_min=params["q_min"],
        q_max=params["q_max"]
    )

    return q_flattened, I_flattened







