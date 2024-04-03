import numpy as np
from scipy.signal import butter, filtfilt


def get_data_with_butterworth_low_pass_filter(data, order, fs, cutoff):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq

    '''
    butter(N, Wn, btype='low', analog=False, output='ba', fs=None)
    ==================================================================
    Butterworth digital and analog filter design.

    Design an Nth-order digital or analog Butterworth filter and return
    the filter coefficients.

    Parameters
    ----------
    N : int
        The order of the filter.
    Wn : array_like
        The critical frequency or frequencies. For lowpass and highpass
        filters, Wn is a scalar; for bandpass and bandstop filters,
        Wn is a length-2 sequence.

        For a Butterworth filter, this is the point at which the gain
        drops to 1/sqrt(2) that of the passband (the "-3 dB point").

        For digital filters, `Wn` are in the same units as `fs`.  By default,
        `fs` is 2 half-cycles/sample, so these are normalized from 0 to 1,
        where 1 is the Nyquist frequency. (`Wn` is thus in
        half-cycles / sample.)
    '''
    b, a = butter(order, normal_cutoff)

    '''
    def filtfilt(b, a, x, axis=-1, padtype='odd', padlen=None, method='pad', irlen=None)
    ==================================================================
    Apply a digital filter forward and backward to a signal.

    This function applies a linear digital filter twice, once forward and
    once backwards.  The combined filter has zero phase and a filter order
    twice that of the original.

    The function provides options for handling the edges of the signal.

    The function `sosfiltfilt` (and filter design using ``output='sos'``)
    should be preferred over `filtfilt` for most filtering tasks, as
    second-order sections have fewer numerical problems.

    Parameters
    ----------
    b : (N,) array_like
        The numerator coefficient vector of the filter.
    a : (N,) array_like
        The denominator coefficient vector of the filter.  If ``a[0]``
        is not 1, then both `a` and `b` are normalized by ``a[0]``.
    x : array_like
        The array of data to be filtered.
    padlen : int or None, optional
        The number of elements by which to extend `x` at both ends of
        `axis` before applying the filter.  This value must be less than
        ``x.shape[axis] - 1``.  ``padlen=0`` implies no padding.
        The default value is ``3 * max(len(a), len(b))``.
    '''
    y_np = filtfilt(b, a, data, padlen=len(data) - 1)

    y_list = [round(y, 2) for y in y_np]
    y_np = np.array(y_list)

    return y_np
