import os
import csv
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output


import G_approximation as G_approx
import ipywidgets as widgets
from itertools import combinations
import matplotlib.pyplot as plt
import save_and_load as csv_man

def process_simulation_pairs_and_fit(log_file, results_folder, fitting_parameters):
    """
    Reads the simulation log CSV file, groups simulations that have identical
    fixed simulation parameters (all columns except for sigma_x, sigma_y, and filename),
    and for each group that has exactly two files (a pair) where sigma_x and/or sigma_y differ,
    loads the two files, runs the fit using G_approximation.G_fit, and collects all fixed simulation parameters.
    
    This version creates a string group key to avoid grouping issues.
    
    Returns a list of dictionaries (one per pair) with keys:
      - 'fixed_params': dict of fixed simulation parameters.
      - 'filename1', 'filename2': filenames.
      - 'q1', 'I1', 'q2', 'I2': data arrays.
      - 'fit_results': dict with fit output (including 'optimal_parameters').
    """
    df = pd.read_csv(log_file)
    exclude_cols = ['sigma_x', 'sigma_y', 'filename']
    group_cols = [col for col in df.columns if col not in exclude_cols]
    # Create a group key as a concatenated string.
    df['group_key'] = df[group_cols].astype(str).agg("_".join, axis=1)
    grouped = df.groupby("group_key")
    
    pairs_data = []
    for group_key, group in grouped:
        # Ensure there are at least 2 files
        if len(group) < 2:
            print(f"Group {group_key} does not have at least 2 files (n={len(group)}); skipping.")
            continue

        # Check if the sigma values across the group are all identical
        if group['sigma_x'].nunique() <= 1 and group['sigma_y'].nunique() <= 1:
            print("Group", group_key, "has identical sigma values; skipping.")
            continue

        # If exactly 2 files, process the single pair
        if len(group) == 2:
            fixed_params = group.iloc[0].drop(labels=exclude_cols + ['group_key']).to_dict()
            filenames = group['filename'].tolist()
            filepath1 = os.path.join(results_folder, filenames[0])
            filepath2 = os.path.join(results_folder, filenames[1])
            try:
                q1, I1 = csv_man.read_q_I_from_csv(filepath1)
                q2, I2 = csv_man.read_q_I_from_csv(filepath2)
                
                fit_results, q1_masked, logdI, G_final, G_initial, G_g0g1_fit = G_approx.G_fit(q1, I1, q2, I2, **fitting_parameters)
                
                
                if fitting_parameters["plot_fitting_curve"]:
                    G_approx.plot_G_function_fits (q1_masked, logdI, G_final,G_initial, G_g0g1_fit, fit_results, filename1 = filenames[0], filename2=filenames[1])   
            except Exception as e:
                print(f"Error processing group {group_key}: {e}")
                continue
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
            # For groups larger than 2, iterate over all unique pairs
            for (idx1, row1), (idx2, row2) in combinations(group.iterrows(), 2):
                fixed_params = row1.drop(labels=exclude_cols + ['group_key']).to_dict()
                filenames = [row1['filename'], row2['filename']]
                filepath1 = os.path.join(results_folder, filenames[0])
                filepath2 = os.path.join(results_folder, filenames[1])
                try:
                    q1, I1 = csv_man.read_q_I_from_csv(filepath1)
                    q2, I2 = csv_man.read_q_I_from_csv(filepath2)
                    fit_results, q1_masked, logdI, G_final, G_initial, G_g0g1_fit = G_approx.G_fit(q1, I1, q2, I2, **fitting_parameters)
                   
                    if fitting_parameters["plot_fitting_curve"]:
                        G_approx.plot_G_function_fits (q1_masked, logdI, G_final,G_initial,G_g0g1_fit, fit_results, filename1 = filenames[0], filename2=filenames[1])  
                except Exception as e:
                    print(f"Error processing group {group_key} for pair {filenames}: {e}")
                    continue
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
    return pairs_data   

def save_pairs_results_to_csv(pairs_data, fitting_parameters, output_csv_filename):
    """
    Saves the pairs_data (a list of dictionaries produced by process_simulation_pairs_and_fit)
    along with the fitting_parameters used, into an external CSV file.
    
    The CSV file will start with comment lines (prefixed with "#") containing the fitting parameters.
    Then, after a blank line, a table is written with one row per pair containing:
      - Fixed simulation parameters.
      - The filenames.
      - The fitted optimal parameters.
    
    Parameters
    ----------
    pairs_data : list of dict
        List returned by process_simulation_pairs_and_fit.
    fitting_parameters : dict
        The dictionary of fitting parameters used.
    output_csv_filename : str
        Path to the output CSV file.
    """
    # Determine all keys from fixed_params and from the optimal_parameters in fit_results.
    fixed_keys = set()
    fit_keys = set()
    for pair in pairs_data:
        fixed = pair.get("fixed_params", {})
        fixed_keys.update(fixed.keys())
        fit_results = pair.get("fit_results")
        if fit_results is not None and "optimal_parameters" in fit_results:
            fit_keys.update(fit_results["optimal_parameters"].keys())
    fixed_keys = sorted(fixed_keys)
    fit_keys = sorted(fit_keys)
    
    # Build the header row for the table.
    header = fixed_keys + ["filename1", "filename2"] + fit_keys
    
    # Build table rows.
    table_rows = []
    for pair in pairs_data:
        row = {}
        fixed = pair.get("fixed_params", {})
        for key in fixed_keys:
            row[key] = fixed.get(key, "")
        row["filename1"] = pair.get("filename1", "")
        row["filename2"] = pair.get("filename2", "")
        row["Ag0"] = pair.get("Ag0","")
        row["Ag1"] = pair.get("Ag1","")
        
        fit_results = pair.get("fit_results")
        if fit_results is not None and "optimal_parameters" in fit_results:
            opt_params = fit_results["optimal_parameters"]
            for key in fit_keys:
                row[key] = opt_params.get(key, "")
        else:
            for key in fit_keys:
                row[key] = ""
        table_rows.append(row)
    
    # Write to CSV.
    with open(output_csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write the fitting parameters as commented header lines.
        writer.writerow(["# Fitting Parameters Used:"])
        for key, value in fitting_parameters.items():
            writer.writerow([f"# {key}: {value}"])
        writer.writerow([])  # Blank line.
        # Write the table header.
        writer.writerow(header)
        # Write each row.
        for row in table_rows:
            writer.writerow([row.get(col, "") for col in header])
    print(f"Results saved to {output_csv_filename}")

def run_fitting_ui(log_file, results_folder):
    # Default fitting parameters.
    state = {"pairs": None}
    default_params = {
        "q_min": 0.0,
        "q_max": 0.2,
        "form_factor_name": "guinier_ff",
        "f2_initial": 0,
        "f2_min": -1,
        "f2_max": 1,
        "f2_free": False,
        "rg_initial": 2,
        "rg_min": 0.5,
        "rg_max": 2.5,
        "rg_free": True,
        "perform_guinier_estimation": False,
        "var_initial": 0.001/4,
        "var_min": 0.000005,
        "var_max": 0.5,
        "var_free": True,
        "A_initial": 0.1,
        "A_min": -1,
        "A_max": 0,
        "A_free": True,
        "auto_init_A": True,
        "plot_fitting_curve": False,
        "auto_set_parameters": False,
        "auto_rg_bound_percent": 0.2
    }
    # Create widgets for each fitting parameter.
    param_widgets = {}
    for key, val in default_params.items():
        if isinstance(val, bool):
            param_widgets[key] = widgets.Checkbox(value=val, description=key)
        else:
            param_widgets[key] = widgets.Text(value=str(val), description=key)
    
    # Dropdown for selecting the x-axis simulation parameter.
    xaxis_dropdown = widgets.Dropdown(
        options=["rg", "variance", "f2_initial"],
        value="variance",
        description="X-axis:"
    )
    
    run_button = widgets.Button(description="Run Fitting and Plot", button_style="success")
    output_area = widgets.Output()
    
    def on_run_button_clicked(b):
        with output_area:
            clear_output(wait=True)
            # Build fitting_params from widget values.
            fitting_params = {}
            for key, widget in param_widgets.items():
                default_val = default_params[key]
                new_val = widget.value
                if isinstance(default_val, bool):
                    fitting_params[key] = new_val.strip().lower() == "true" if isinstance(new_val, str) else bool(new_val)
                else:
                    try:
                        fitting_params[key] = float(new_val)
                    except ValueError:
                        fitting_params[key] = new_val
            # print("Using fitting parameters:", fitting_params)
            
            try:
                pairs = process_simulation_pairs_and_fit(log_file, results_folder, fitting_params)
            except Exception as e:
                print("Error processing simulation pairs:", e)
                return
            
            # Aggregate data for plotting.
            x_param = xaxis_dropdown.value
            x_values = []
            y_rg = []    # simulation rg / fitted rg
            y_var = []   # simulation variance / fitted variance
            y_f2 = []    # simulation f2 / fitted f2
            for pair in pairs:
                fixed = pair["fixed_params"]
                if x_param not in fixed:
                    continue
                try:
                    x_val = float(fixed[x_param])
                except Exception:
                    continue
                x_values.append(x_val)
                sim_rg = float(fixed.get("rg", np.nan))
                sim_var = float(fixed.get("variance", np.nan))
                sim_f2 = float(fixed.get("f2_initial", 0))
                fit_opt = pair["fit_results"]["optimal_parameters"]
                emp_var = float(fixed.get("empirical_var", 0))
                emp_mean = float(fixed.get("empirical_mean", 0))
                #print ("fit_opt[rg_fit]",fit_opt["rg_fit"])
                fitted_rg = float(fit_opt.get("rg_fit", np.nan))
                fitted_var = float(fit_opt.get("var_fit", np.nan))
                fitted_f2 = float(fit_opt.get("f2_fit", np.nan))
                
                rg_ratio = (fitted_rg -emp_mean) / emp_mean if emp_mean != 0 else np.nan
                var_ratio =(fitted_var - emp_var) / emp_var if emp_var != 0 else np.nan
                f2_ratio = (fitted_f2-sim_f2) / sim_f2 if sim_f2 != 0 else np.nan
                
                y_rg.append(rg_ratio)
                y_var.append(var_ratio)
                y_f2.append(f2_ratio)
            
            # Create one figure with three subplots.
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(4, 6))
            ax1.scatter(x_values, y_rg, color="blue")
            ax1.set_title("(Rg_fit-Rg_sim) / Rg_sim")
            ax1.set_xlabel(x_param)
            ax1.set_ylabel("Rg ratio")
            
            ax2.scatter(x_values, y_var, color="green")
            ax2.set_title("(Var_fit-Var_sim) / Var_sim")
            ax2.set_xlabel(x_param)
            ax2.set_ylabel("Variance ratio")
            
            ax3.scatter(x_values, y_f2, color="red")
            ax3.set_title("(f2_fit-f2_sim) / f2_sim")
            ax3.set_xlabel(x_param)
            ax3.set_ylabel("f2 ratio")
            
            fig.tight_layout()
            display(fig)
            plt.close(fig)
            
            state["pairs"] = pairs
            # Save the pairs results to an external CSV file.
            output_csv = os.path.join(results_folder, "fitting_results.csv")
            try:
                save_pairs_results_to_csv(pairs, fitting_params, output_csv)
            except Exception as e:
                print("Error saving results to CSV:", e)
        
    run_button.on_click(on_run_button_clicked)
    
    ui_left = widgets.VBox(list(param_widgets.values()) + [xaxis_dropdown, run_button])
    ui = widgets.HBox([ui_left, output_area])
    display(ui)
    
    def get_pairs():
        return state["pairs"]
    
    return get_pairs
    
def parse_value(s):
    """
    Attempts to parse a string into a boolean or float.
    If neither conversion works, returns the string as-is.
    """
    s = s.strip()
    # Check for booleans first.
    if s.lower() == "true":
        return True
    elif s.lower() == "false":
        return False
    # Try converting to a float.
    try:
        return float(s)
    except ValueError:
        return s

def read_fitting_results_csv(csv_filename):
    """
    Reads the fitting_results CSV file produced by the saving code.
    
    The file is expected to have:
      - A header block of commented lines that contain the fitting parameters.
      - A blank line.
      - A table header and rows containing the simulation pairing results.
      
    The fitting parameters are parsed using parse_value so that booleans,
    floats, and strings are recovered correctly.
    
    Parameters
    ----------
    csv_filename : str
        Path to the CSV file.
        
    Returns
    -------
    fitting_parameters : dict
        Dictionary of fitting parameters extracted from the header.
    df : pandas.DataFrame
        DataFrame containing the table data.
    """
    fitting_parameters = {}
    header = None
    table_rows = []
    
    with open(csv_filename, "r", newline="") as csvfile:
        reader = csv.reader(csvfile)
        
        # Read and parse the commented header lines.
        for row in reader:
            if row and row[0].strip().startswith("#"):
                line = row[0].strip()
                # Skip the title line.
                if line.startswith("# Fitting Parameters Used:"):
                    continue
                else:
                    # Remove '#' and any whitespace, then split at the first colon.
                    line_content = line.lstrip("#").strip()
                    parts = line_content.split(":", 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip()
                        fitting_parameters[key] = parse_value(value)
            else:
                # Skip blank lines.
                if not any(cell.strip() for cell in row):
                    continue
                # The first non-comment, non-blank row is assumed to be the table header.
                header = row
                break
        
        # Read the remaining rows as table data.
        for row in reader:
            if row and any(cell.strip() for cell in row):
                table_rows.append(row)
    
    # Create a DataFrame from the table rows.
    df = pd.DataFrame(table_rows, columns=header)
    return fitting_parameters, df

def load_and_plot_pair (df_pairs, selected_index, fitting_params,results_folder):
    row_data = df_pairs.loc[selected_index]
            
    # Extract filenames from the selected row
    fn1 = row_data["filename1"]
    fn2 = row_data["filename2"]

    print(f"Selected row {selected_index}")
    print(f"  Loading File 1: {fn1}")
    print(f"  Loading File 2: {fn2}")
    
    # Read the actual data from the two files
    filepath1 = os.path.join(results_folder, fn1)
    filepath2 = os.path.join(results_folder, fn2)

    q1, I1 = csv_man.read_q_I_from_csv(filepath1)
    q2, I2 = csv_man.read_q_I_from_csv(filepath2)
    
    # Turn on plotting in the fitting parameters
    fitting_params["plot_fitting_curve"] = True

    # Perform the fit
    fit_results, q1_masked, logdI, G_final, G_initial, G_g0g1_fit  = G_approx.G_fit(q1, I1, q2, I2, **fitting_params)
    G_approx.plot_G_function_fits(q1_masked,logdI,G_final,G_initial,G_g0g1_fit,fit_results,fn1,fn2)
