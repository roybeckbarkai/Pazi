import os

import itertools
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import analytical_simulation_2d
import binning
import G_approximation

import matplotlib.pyplot as plt
import save_and_load as csv_man

def process_simulation_pairs_and_fit(log_file, results_folder, fitting_parameters):
    """
    Reads the simulation log CSV file, groups simulations that have identical
    fixed simulation parameters (all columns except for sigma_x, sigma_y, and filename),
    and for each group that has exactly two files (i.e. a pair) where sigma_x and/or sigma_y differ,
    loads the two files, runs the fit using G_approximation.G_fit, and collects all fixed simulation parameters.
    
    Returns a list of dictionaries, one per pair, with keys:
      - 'fixed_params': dict of all fixed simulation parameters,
      - 'filename1', 'filename2': simulation filenames,
      - 'q1', 'I1': data arrays from file 1,
      - 'q2', 'I2': data arrays from file 2,
      - 'fit_results': dict with fit output (including 'optimal_parameters').
    """
    df = pd.read_csv(log_file)
    # Use all columns except sigma_x, sigma_y, filename for grouping.
    exclude_cols = ['sigma_x', 'sigma_y', 'filename']
    group_cols = [col for col in df.columns if col not in exclude_cols]
    grouped = df.groupby(group_cols)
    
    pairs_data = []
    for group_keys, group in grouped:
        if len(group) == 2:
            if group['sigma_x'].nunique() > 1 or group['sigma_y'].nunique() > 1:
                fixed_params = group.iloc[0].drop(labels=exclude_cols).to_dict()
                filenames = group['filename'].tolist()
                filepath1 = os.path.join(results_folder, filenames[0])
                filepath2 = os.path.join(results_folder, filenames[1])
                try:
                    q1, I1 = csv_man.read_q_I_from_csv(filepath1)
                    q2, I2 = csv_man.read_q_I_from_csv(filepath2)
                    fit_results = G_approximation.G_fit(q1, I1, q2, I2, **fitting_parameters)
                except Exception as e:
                    print(f"Error processing group {group_keys}: {e}")
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
                print("Group", group_keys, "has identical sigma values; skipping.")
        else:
            print("Group", group_keys, f"does not have exactly 2 files (n={len(group)}); skipping.")
    return pairs_data

class FittingUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Simulation Fitting and Analysis")
        self.geometry("1200x800")
        
        # Default fitting parameters dictionary.
        self.fitting_parameters = {
            "q_min": None,
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
            "A_initial": -4.483e-5* 3/(2*4*(1+0.01)),
            "A_min": -1,
            "A_max": 0,
            "A_free": True,
            "plot_fitting_curve": True,
            "auto_set_parameters": True,
            "auto_rg_bound_percent": 0.2
        }
        # Create UI controls for fitting parameters.
        self.param_entries = {}
        row = 0
        tk.Label(self, text="Fitting Parameters", font=("Arial", 12, "bold")).grid(row=row, column=0, columnspan=2, pady=(10,5))
        row += 1
        for key, val in self.fitting_parameters.items():
            tk.Label(self, text=key).grid(row=row, column=0, sticky="w", padx=5)
            entry = tk.Entry(self, width=15)
            entry.insert(0, str(val))
            entry.grid(row=row, column=1, sticky="w", padx=5)
            self.param_entries[key] = entry
            row += 1
        
        # Dropdown menu for selecting x-axis simulation parameter.
        tk.Label(self, text="X-axis parameter", font=("Arial", 10)).grid(row=row, column=0, sticky="w", padx=5, pady=5)
        self.xaxis_var = tk.StringVar(self)
        # The typical simulation fixed parameters available: 'rg', 'variance', and optionally 'f2_initial'.
        xaxis_options = ["rg", "variance", "f2_initial"]
        self.xaxis_var.set("rg")
        self.xaxis_menu = ttk.Combobox(self, textvariable=self.xaxis_var, values=xaxis_options, state="readonly", width=12)
        self.xaxis_menu.grid(row=row, column=1, sticky="w", padx=5, pady=5)
        row += 1
        
        # Button to run the fitting.
        self.run_button = tk.Button(self, text="Run Fitting and Plot", command=self.run_fitting, bg="lightblue")
        self.run_button.grid(row=row, column=0, columnspan=2, pady=10)
        row += 1
        
        # Frame for Matplotlib figures.
        self.fig = Figure(figsize=(8, 6))
        self.ax_rg = self.fig.add_subplot(311)
        self.ax_var = self.fig.add_subplot(312)
        self.ax_f2 = self.fig.add_subplot(313)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().grid(row=0, column=2, rowspan=row, padx=10, pady=10)
        
        # Paths for simulation log and results folder.
        self.log_file = "simulation_results/simulation_log.csv"
        self.results_folder = "simulation_results"
    
    def run_fitting(self):
        # Update fitting parameters from UI entries.
        for key, entry in self.param_entries.items():
            try:
                # Convert to float if possible; otherwise, keep as string.
                val = float(entry.get())
            except ValueError:
                val = entry.get()
            self.fitting_parameters[key] = val
        
        # Process simulation pairs and run fitting.
        try:
            pairs = process_simulation_pairs_and_fit(self.log_file, self.results_folder, self.fitting_parameters)
        except Exception as e:
            print("Error processing simulation pairs:", e)
            return
        
        # Prepare lists for plotting.
        # x_values: chosen simulation fixed parameter from each pair.
        # y_rg: simulation rg / fitted rg, y_var: simulation variance / fitted variance, y_f2: simulation f2 / fitted f2.
        x_param = self.xaxis_var.get()  # e.g., "rg", "variance", or "f2_initial"
        x_values = []
        y_rg = []
        y_var = []
        y_f2 = []
        
        # Loop over pairs.
        for pair in pairs:
            fixed = pair["fixed_params"]
            # If the selected x-axis parameter is not present, skip.
            if x_param not in fixed:
                continue
            try:
                x_val = float(fixed[x_param])
            except Exception:
                continue
            x_values.append(x_val)
            # Simulation values.
            sim_rg = float(fixed.get("rg", np.nan))
            sim_var = float(fixed.get("variance", np.nan))
            sim_f2 = float(fixed.get("f2_initial", 0))  # default 0 if not provided
            
            # Fitted parameters (from fit_results dictionary).
            fit_opt = pair["fit_results"]["optimal_parameters"]
            fitted_rg = float(fit_opt.get("rg_fit", np.nan))
            fitted_var = float(fit_opt.get("var_fit", np.nan))
            fitted_f2 = float(fit_opt.get("f2_fit", np.nan))
            
            # Calculate ratios; avoid division by zero.
            rg_ratio = sim_rg / fitted_rg if fitted_rg != 0 else np.nan
            var_ratio = sim_var / fitted_var if fitted_var != 0 else np.nan
            f2_ratio = sim_f2 / fitted_f2 if fitted_f2 != 0 else np.nan
            
            y_rg.append(rg_ratio)
            y_var.append(var_ratio)
            y_f2.append(f2_ratio)
        
        # Clear axes.
        self.ax_rg.cla()
        self.ax_var.cla()
        self.ax_f2.cla()
        
        # Plot the ratios vs. the selected x-axis parameter.
        self.ax_rg.scatter(x_values, y_rg, color="blue")
        self.ax_rg.set_title("Simulation Rg / Fitted Rg")
        self.ax_rg.set_xlabel(x_param)
        self.ax_rg.set_ylabel("Rg ratio")
        
        self.ax_var.scatter(x_values, y_var, color="green")
        self.ax_var.set_title("Simulation Variance / Fitted Variance")
        self.ax_var.set_xlabel(x_param)
        self.ax_var.set_ylabel("Variance ratio")
        
        self.ax_f2.scatter(x_values, y_f2, color="red")
        self.ax_f2.set_title("Simulation f2 / Fitted f2")
        self.ax_f2.set_xlabel(x_param)
        self.ax_f2.set_ylabel("f2 ratio")
        
        self.fig.tight_layout()
        self.canvas.draw()

if __name__ == "__main__":
    app = FittingUI()
    app.mainloop()