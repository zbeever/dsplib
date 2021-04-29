import numpy as np
from numba import njit
import tqdm

@njit
def extend_array(x, V=0):
    '''
    Extends an N-point signal to the an 2^V-point signal via zero padding

    Parameters
    ----------
    x : float[N]
        The signal to extend.

    V : int (optional)
        The power of 2 to extend the signal to. Defaults to 0, in which case the function
        pads the signal to the nearest, larger power of 2.

    Returns
    -------
    x_prime : float[2^V]
        The padded signal.

    v : int
        The power of 2 used.
    '''

    N = len(x) # Length of the original signal
    v = int(max(np.ceil(np.log2(N)), V)) # The next largest power of 2 (or, if it is greater, V)

    x_prime = np.zeros(int(2**v), dtype=np.complex128)
    x_prime[:N] = x

    return x_prime, v


@njit
def convolve(a, b):
    '''
    Convolves an M-point signal with an N-point signal. Assumes both signals start at 0.

    Parameters
    ----------
    a : float[N]
        A signal to convolve.

    b : float[M]
        A signal to convolve.

    Returns
    -------
    convolved_sig : float[N + M - 1]
        The convolved signal.
    '''

    M = len(a)
    N = len(b)

    ns = np.arange(0, N + M - 1)
    convolved_sig = np.zeros(N + M - 1)

    for n in ns:
        low = max(0, n - (N - 1))
        high = min(n, M - 1)
        ms = np.arange(low, high + 1)
        convolved_sig[n] = np.sum(a[ms] * b[n - ms])

    return convolved_sig


@njit
def upsample(x, n=2):
    '''
    Upsamples a signal by n, inserting 0s between each sample.

    Parameters
    ----------
    x : float[N]
        The signal.

    n : int (optional)
        The upsampling rate. Defaults to 2.

    Returns
    -------
    y : float[nN]
        The upsampled signal.
    '''

    y = np.zeros(n * len(x))
    y[::n] = x
    return y


@njit
def downsample(x, n=2):
    '''
    Downsamples the signal by n.

    Parameters
    ----------
    x : float[N]
        The signal.

    n : int (optional)
        The downsampling rate. Defaults to 2.

    Returns
    -------
    y : float[N/n]
        The downsampled signal.
    '''

    return x[::2]
