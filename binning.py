import numpy as np
import pandas as pd

def bin_and_match_saxs_data(q1, I1, q2, I2, bins):
    """
    Bin two SAXS data sets onto a common q grid and find the overlapping region.

    Parameters:
        q1, I1 (array-like): q-values and intensities for dataset 1
        q2, I2 (array-like): q-values and intensities for dataset 2
        bins (int or array-like): Number of bins or bin edges for common binning

    Returns:
        DataFrame: Binned q, I1, and I2 in the overlapping q-region
    """

    # Define common bin edges based on both q datasets
    bin_edges = np.histogram_bin_edges(np.hstack((q1, q2)), bins=bins)

    # Helper function to bin data
    def bin_data(q, I, bin_edges):
        q = np.asarray(q).squeeze()  # Ensure 1D
        I = np.asarray(I).squeeze()  # Ensure 1D

        if q.ndim != 1 or I.ndim != 1:
            raise ValueError("q and I must be 1D arrays")

        df = pd.DataFrame({"q": q, "I": I})
        df["bin"] = pd.cut(df["q"], bins=bin_edges, labels=False)
        binned = df.groupby("bin").agg({"q": "mean", "I": "mean"}).dropna()

        return binned

    # Bin both datasets
    binned1 = bin_data(q1, I1, bin_edges)
    binned2 = bin_data(q2, I2, bin_edges)

    # Find common q bins
    common_bins = binned1.index.intersection(binned2.index)

    # Extract overlapping region
    overlap1 = binned1.loc[common_bins]
    overlap2 = binned2.loc[common_bins]

    # Combine into a single DataFrame
    result = pd.DataFrame({"q": overlap1["q"], "I1": overlap1["I"], "I2": overlap2["I"]})

    return result

