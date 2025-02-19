import pandas as pd
import os
import fabio
import lmfit
from datetime import datetime

# Save 1D I vs q data as CSV (.dat) files for further analysis
def save_1d_to_csv(output_folder, output_file_name, q, intensity):
    df = pd.DataFrame({'q': q, 'Intensity': intensity})
    df.to_csv(output_folder + output_file_name + ".dat", sep=",", index=False)
    out_name=output_folder + output_file_name + ".dat"
    print(f"2D intensity data flattened to {out_name}")

# Saves 2D Intensities array to .csv file with coma separation
def save_2d_to_csv(output_folder, output_file_name, intensity_array_2d):
    # Save the Intensity array as a CSV file
    os.makedirs(output_folder, exist_ok=True)
    pd.DataFrame(intensity_array_2d).to_csv(output_folder + output_file_name, index=False, header=False)
    print(f"2D intensity picture created and saved to {output_folder+ output_file_name}")

#Converts a .CSV file containing a 2D array of intensities to a .TIF file.
def csv_to_tif(input_folder, input_csv, output_tif):
    # Load the .CSV as a 2D NumPy array
    #data = np.loadtxt(input_csv, delimiter=',')
    data = pd.read_csv(input_folder+input_csv, header=None).values

    # Save the .2D array as a TIFF file using FabIO
    tif_image = fabio.tifimage.TifImage(data=data)
    tif_image.write(output_tif)
    print(f"2D intensity array saved to {input_folder + output_tif}")

# Function to append fitting results to a log file
def save_fit_results_to_log(log_file_name: str, results: lmfit.model.ModelResult): #, chi_sq: float):
    """
    Save fitting results to a .log file (CSV format).

    Parameters:
        log_file_name (str): The name of the log file.
        results (lmfit.model.ModelResult): The fitting results.
        chi_sq (float): The chi-square value of the fit.
    """
    # Ensure the file has a .log extension
    if not log_file_name.endswith(".log"):
        log_file_name += ".log"

    # Check if file exists
    file_exists = os.path.isfile(log_file_name)

    # Open the file in append mode
    with open(log_file_name, "a") as log_file:
        # If file is new, write the header
        if not file_exists:
            log_file.write("# G function fitting Log\n")

        log_file.write(f"\n# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        #log_file.write("rg_fit,var_fit,f2_fit,A_fit,chi_square\n")  # CSV header
        # Write the fitting parameters
        #param_values = [results.params[p].value for p in ["rg_fit", "var_fit", "f2_fit", "A_fit"]]
        #log_file.write(",".join(map(str, param_values)) + f",{chi_sq}\n")
        log_file.write("".join(map(str, results)) + f"\n")
    print(f"Fitting results saved to {log_file_name}")



