
def ClutterRemoval(input_val, axis=0):
    """Clutter removal of the input data.

    Parameters
    ----------
    input_val : ndarray
        Input data.
    axis : int, optional
        Axis along which to perform the clutter removal.

    Returns
    -------
    ndarray
        Clutter removed data.

    """
    
    mean = input_val.mean(axis) # mean over the axis 0 (range) 
    output_val = input_val - mean
    
    return output_val