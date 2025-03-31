import analytical_simulation_2d
import binning
import G_approximation

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import save_and_load as csv_man


def process_simulation_pairs(log_file, results_folder):
    """
    Reads the simulation log CSV file, groups simulations that have identical
    parameters except for sigma_x and sigma_y, and for each group that has exactly
    two files (i.e. a pair), loads the two files and collects all the fixed simulation
    parameters for that group.

    Parameters
    ----------
    log_file : str
        Path to the simulation log CSV file.
    results_folder : str
        Folder where the simulation CSV files are stored.

    Returns
    -------
    pairs_data : list of dict
        Each dict corresponds to a pair and has keys:
          - 'fixed_params': a dictionary with all fixed simulation parameters for the group.
          - 'filename1', 'filename2': the filenames.
          - 'q1', 'I1': data from the first file.
          - 'q2', 'I2': data from the second file.
    """
    # Read the log file into a DataFrame.
    df = pd.read_csv(log_file)
    
    # Determine which columns to group by.
    # We exclude "sigma_x", "sigma_y", and "filename" because we allow differences in these.
    exclude_cols = ['sigma_x', 'sigma_y', 'filename']
    group_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Group the DataFrame by these columns.
    grouped = df.groupby(group_cols)
    
    pairs_data = []
    
    # Process each group.
    for group_keys, group in grouped:
        # Only consider groups with exactly two files.
        if len(group) == 2:
            # Check if sigma_x or sigma_y differs.
            if group['sigma_x'].nunique() > 1 or group['sigma_y'].nunique() > 1:
                # Convert group_keys to a dictionary of fixed parameters.
                if isinstance(group_keys, tuple):
                    fixed_params = dict(zip(group_cols, group_keys))
                else:
                    fixed_params = {group_cols[0]: group_keys}
                
                filenames = group['filename'].tolist()
                filepath1 = os.path.join(results_folder, filenames[0])
                filepath2 = os.path.join(results_folder, filenames[1])
                try:
                    q1, I1 = csv_man.read_q_I_from_csv(filepath1)
                    q2, I2 = csv_man.read_q_I_from_csv(filepath2)
                except Exception as e:
                    print(f"Error reading files {filenames[0]} and {filenames[1]}: {e}")
                    continue
                
                 # Run the fitting function on the pair.
                try:
                    fit_results = G_approximation.G_fit(q1, I1, q2, I2, **fitting_parameters)
                except Exception as e:
                    print(f"Error fitting pair ({filenames[0]}, {filenames[1]}): {e}")
                    fit_results = None  # Optionally, you could skip this pair.
                
                pair_dict = {
                    "fixed_params": fixed_params,
                    "filename1": filenames[0],
                    "filename2": filenames[1],
                    "q1": q1,
                    "I1": I1,
                    "q2": q2,
                    "I2": I2,
                    "fit_results": fit_results
                }
                pairs_data.append(pair_dict)
            else:
                print("Group with keys", group_keys, "has identical sigma values; skipping.")
        else:
            print(f"Group with keys {group_keys} does not have exactly 2 files (n={len(group)}); skipping.")
    
    return pairs_data

if __name__ == "__main__":
    # Path to the log file and the folder where simulation files are stored.
        # Define the fitting parameters dictionary.
    fitting_parameters = {
        # q range limits
        "q_min": None,
        "q_max": 0.2,
    
        # f2 input
        "form_factor_name": "guinier_ff",
        "f2_initial": 0,
        "f2_min": -1,
        "f2_max": 1,
        "f2_free": False,
    
        # Rg input
        "rg_initial": 2,
        "rg_min": 0.5,
        "rg_max": 2.5,
        "rg_free": True,
        "perform_guinier_estimation": False,
    
        # Variance input
        "var_initial": 0.001/4,
        "var_min": 0.000005,
        "var_max": 0.5,
        "var_free": True,
    
        # A (scaling factor) input
        "A_initial": -4.483e-5* 3/(2*4*(1+0.01)),
        "A_min": -1,
        "A_max": 0,
        "A_free": True,
    
        # Additional options
        "plot_fitting_curve": False,
        "auto_set_parameters": True,
        "auto_rg_bound_percent": 0.2
    }
    
    
    
    log_file = "simulation_results/simulation_log.csv"
    results_folder = "simulation_results"
    
    pairs = process_simulation_pairs(log_file, results_folder)
    
    # Print a summary for each pair.
    for pair in pairs:
        print("Fixed simulation parameters:", pair["fixed_params"])
        print("  File 1:", pair["filename1"], "Length of q1:", len(pair["q1"]))
        print("  File 2:", pair["filename2"], "Length of q2:", len(pair["q2"]))
        print("-" * 40)