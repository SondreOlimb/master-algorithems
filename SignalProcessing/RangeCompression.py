
import numpy as np
def RangeCompression(adc_data, window_type_1d=None, axis=0 ,fft_size=256):
    """Range processing of the input data.

    Parameters
    ----------
    adc_data : ndarray
        Input data.
    window_type_1d : str, optional
        Window type for 1D FFT.
    axis : int, optional
        Axis along which to perform the range processing.

    Returns
    -------
    ndarray
        Range processed data.

    """
    if window_type_1d is not None:
        window = np.hanning(adc_data.shape[axis])
        adc_data = adc_data * window

    return np.fft.fft(adc_data, axis=axis,n=fft_size)