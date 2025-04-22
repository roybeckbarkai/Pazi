import os
import itertools
import csv
import numpy as np

# Import your simulation and CSV management modules
import analytical_simulation_2d
import matplotlib.pyplot as plt
import numpy as np
import save_and_load as csv_man
import binning

def main():
    # Default simulation parameters (as provided)
    sim_params_default = {
        # Detector and source parameters:
        "px_number": [500, 500],
        "px_size": 0.075,           # mm
        "wavelength": 0.154,        # nm
        # Sample parameters:
        "form_factor_name": "guinier_ff",
        #"form_factor_name": "ext_guinier_ff",
        "rg": 4,                   # in nm
        "variance": 0.01,
        "sigma_x": 0.01,           # in pixels
        "sigma_y": 0.01,           # in pixels
        "sample_detector_distance": 1500,  # in mm
        # Flattening parameters:
        "normalization_on": True,
        "return_unique": True,
        "q_min": 0.0001,
        "q_max": 0.25,
        "binning": False,
        "bins_number": 1000 
    }
    
    # Define the parameters that you want to vary and their values.
    # For example, vary 'rg' and 'variance'. You can add more keys if needed.
    vary_params = {
        "rg": [1, 4 ,6 ],          # example values for rg (in nm)
        "variance": [0.02,   0.2, 0.6],
        "sigma_x":[1 , 5],
        "sigma_y":[1]
    }
    
    # Designated folder to save simulation results.
    save_folder = "simulation_Emp_mean_apr21"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    # Prepare a log file to record simulation parameters and the filename for each simulation.
    log_filename = os.path.join(save_folder, "simulation_log.csv")
    
    # Create a list of keys for the log. 
    # Include "empirical_var" and "empirical_Rg" so we can record the simulation result.
    log_fields = sorted(
        set(
            list(sim_params_default.keys()) 
            + list(vary_params.keys()) 
            + ["filename", "empirical_var", "empirical_mean"]
        )
    )
    
    log_rows = []
    # Create all permutations (combinations) of the varied parameters.
    vary_keys = list(vary_params.keys())
    vary_values_product = list(itertools.product(*(vary_params[k] for k in vary_keys)))
    
    sim_count = 0
    for combo in vary_values_product:
        # Start with a copy of the default simulation parameters.
        sim_params = sim_params_default.copy()
        # Update with the current combination of varied parameters.
        for key, value in zip(vary_keys, combo):
            sim_params[key] = value
        
        # Run the simulation using your provided simulation function.
        # This function returns q and I - 2D data.
        q, I, empirical_var, empirical_mean = analytical_simulation_2d.single_analytical_simulation_flattened(sim_params)
        
        
        if sim_params["binning"]:
              binned_data = binning.bin_and_match_saxs_data(q, I, q, I, sim_params["bins_number"])
              q = binned_data["q"].values
              I = binned_data["I1"].values
        else:
            q=np.array(q).squeeze()
            I=np.array(I).squeeze()
      
        
        
        # Create a unique filename. Here we use the simulation count and the varied parameters.
        filename = f"sim_{sim_count:03d}"
        for key in vary_keys:
            # Append the parameter name and value (formatted to 3 decimals if float)
            param_val = sim_params[key]
            if isinstance(param_val, float):
                filename += f"_{key}{param_val:.3f}"
            else:
                filename += f"_{key}{param_val}"
        filename += ".csv"
        filepath = os.path.join(save_folder, filename)
        
        # Save the (q, I) data using your csv_man module.
        csv_man.save_q_I_to_csv(q, I, filepath)
        
        # Record the simulation parameters along with the filename in the log.
        log_entry = {}
        for key in log_fields:
            if key == "filename":
                log_entry[key] = filename
            elif key == "empirical_var":
                # Record the computed empirical variance
                log_entry[key] = empirical_var
            elif key == "empirical_mean":
                # Record the computed empirical mean
                log_entry[key] = empirical_mean
            elif key in sim_params:
                log_entry[key] = sim_params[key]
            else:
                log_entry[key] = ""  # empty if not defined
        log_rows.append(log_entry)
        
        sim_count += 1
    
    # Write the log file as a CSV.
    # Check if the log file already exists.
    file_exists = os.path.exists(log_filename)
    with open(log_filename, mode="a", newline="") as logfile:
        writer = csv.DictWriter(logfile, fieldnames=log_fields)
        # Write the header only if the file is being created for the first time.
        if not file_exists:
            writer.writeheader()
        for row in log_rows:
            writer.writerow(row)
    
    print(f"Simulation complete: {sim_count} files saved in '{save_folder}'.")
    print(f"Log file saved as '{log_filename}'.")

if __name__ == "__main__":
    main()