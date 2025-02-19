import numpy as np

def save_q_I_to_csv(q, I, filename="data.csv"):
    """
    Save two arrays (q and I) to a CSV file as two columns.
    
    Parameters
    ----------
    q : array-like
        Array of q values.
    I : array-like
        Array of I values.
    filename : str, optional
        The name of the CSV file to save (default is "data.csv").
    
    Raises
    ------
    ValueError
        If the lengths of q and I do not match.
    """
    # Convert to numpy arrays in case they're not already
    q = np.array(q)
    I = np.array(I)
    
    # Check that arrays have the same length
    if q.shape[0] != I.shape[0]:
        raise ValueError("q and I arrays must have the same length.")
    
    # Combine arrays column-wise
    data = np.column_stack((q, I))
    
    # Save to CSV with a header
    np.savetxt(filename, data, delimiter=",", header="q,I", comments="")
    
    print(f"Data successfully saved to {filename}.")

def read_q_I_from_csv(filename="data.csv"):
    """
    Read a CSV file and return the q and I arrays.
    
    The CSV file should have two columns (with a header, e.g., "q,I").
    
    Parameters
    ----------
    filename : str, optional
        The name of the CSV file to read (default is "data.csv").
    
    Returns
    -------
    q : numpy.ndarray
        Array of q values.
    I : numpy.ndarray
        Array of I values.
    
    Raises
    ------
    IOError
        If the file cannot be read.
    ValueError
        If the CSV file does not have two columns.
    """
    try:
        # Load data, skipping the header row
        data = np.loadtxt(filename, delimiter=",", skiprows=1)
    except Exception as e:
        raise IOError(f"Error reading file {filename}: {e}")
    
    # Handle the case where there's only one line of data (1D array)
    if data.ndim == 1:
        if data.size != 2:
            raise ValueError("CSV file must contain exactly two columns per row.")
        q, I = np.array([data[0]]), np.array([data[1]])
    else:
        if data.shape[1] != 2:
            raise ValueError("CSV file must have exactly two columns.")
        q, I = data[:, 0], data[:, 1]
    
    return q, I