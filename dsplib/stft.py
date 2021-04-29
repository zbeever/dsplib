import numpy as np
from numba import njit
import tqdm

from dislib.utils import *
from dislib.fft import *

def rectangular_window(N, P):
    '''
    Generates a rectangular analysis window and the weight to use when performing
    constant overlap add reconstruction.

    Parameters
    ----------
    N : int
        The length of the window.

    P : int
        Parameter describing the expected window overlap. 0 is no overlap, 1 is half overlap, 2 is quarter overlap, etc. 

    Returns
    -------
    w : float[N]
        The window.

    f : float
        The weight to use in constant overlap add reconstruction.
    '''

    return np.ones(N), 0.5**P


def triangular_window(N, P):
    '''
    Generates a triangular analysis window and the weight to use when performing
    constant overlap add reconstruction.

    Parameters
    ----------
    N : int
        The length of the window.

    P : int
        Parameter describing the expected window overlap. 0 is no overlap, 1 is half overlap, 2 is quarter overlap, etc. 

    Returns
    -------
    w : float[N]
        The window.

    f : float
        The weight to use in constant overlap add reconstruction.
    '''

    ns = np.arange(N)
    return 1 - np.abs((ns - N / 2) / (N / 2)), 0.5**(P - 1)


def hann_window(N, P):
    '''
    Generates a Hann analysis window and the weight to use when performing
    constant overlap add reconstruction.

    Parameters
    ----------
    N : int
        The length of the window.

    P : int
        Parameter describing the expected window overlap. 0 is no overlap, 1 is half overlap, 2 is quarter overlap, etc. 

    Returns
    -------
    w : float[N]
        The window.

    f : float
        The weight to use in constant overlap add reconstruction.
    '''

    ns = np.arange(N)
    return np.sin(np.pi * ns / N)**2, 0.5**(P - 1)


def hamming_window(a0, N, P):
    '''
    Generates a Hamming analysis window and the weight to use when performing
    constant overlap add reconstruction.

    Parameters
    ----------
    N : int
        The length of the window.

    P : int
        Parameter describing the expected window overlap. 0 is no overlap, 1 is half overlap, 2 is quarter overlap, etc. 

    Returns
    -------
    w : float[N]
        The window.

    f : float
        The weight to use in constant overlap add reconstruction.
    '''

    a0 = min(a0, 0.5)
    ns = np.arange(N)
    return a0 - (1 - a0) * np.cos(2 * np.pi * ns / N), 0.5**P / (a0)


@njit
def get_frame(x, w, n):
    '''
    Given a signal, a window, and a starting index, returns the associated frame.

    Parameters
    ----------
    x : float[N]
        The signal. Indexing is assumed to start at 0.

    w : float[M]
        The window. Indexing is assumed to start at 0.

    n : int
        The index of the signal at which to start the frame.

    Returns
    -------
    frame : float[M]
        The windowed signal.
    '''

    N = len(w) # The length of the analysis window

    # Clip the first and last index of x to be within the range of x
    first_index = min(n, len(x))
    last_index = min(n + N, len(x))

    # Store a frame-length chunk of x starting at n
    frame = np.zeros(N, dtype=np.complex128)
    frame[:last_index - first_index] = x[first_index:last_index]

    # Multiply the above segment of x by w --- this is the frame
    frame *= w

    return frame


def stft(x, w, L, M, quiet=False):
    '''
    The discrete short time Fourier transform. This allows a time-frequency analysis
    of a signal.

    Parameters
    ----------
    x : float[N]
        The signal to analyze.

    w : float[R]
        The analysis window.

    L : int
        The temporal sampling factor (the distance between successive frames).

    M : int
        The length of the FFT.

    quiet : bool (optional)
        Whether the progress bar should be suppressed. Defaults to False.

    Returns
    -------
    X : float[S, M]
        Returns an array storing the complex STFT of the signal; each column is the
        FFT of a particular frame.
    '''

    if len(x) % L == 0:
        frames = int(len(x) // L)
    else:
        frames = int(len(x) // L + 1)
    X = np.zeros((frames, M), dtype=np.complex128)

    for n in tqdm.tqdm(range(frames), disable=quiet):
        frame = get_frame(x, w, n * L)
        X[n, :] = fft(frame, M) # Sample the discrete frequencies by M

    return X


def istft(X, f, L, truncate=None, quiet=False):
    '''
    The inverse discrete short time Fourier transform. This uses constant overlap add.
    If you are modifying a signal's STFT, the probability that this new STFT has a proper
    inverse is 0. That is, overlap add is a heuristic reconstruction method.

    Parameters
    ----------
    X : float[S, M]
        The STFT to invert.

    f : float
        The weight to use in constant overlap add.

    L : int
        The temporal sampling factor (the distance between successive frames in samples).

    truncate : int (optional)
        The number of samples the inverse FFT of each frame should be truncated to. If not specified,
        defaults to None, which has no truncation.

    quiet : bool (optional)
        Whether the progress bar should be suppressed. Defaults to False.

    Returns
    -------
    x : float[N]
        The (approximate) signal corresponding to the input STFT.
    '''

    N = np.shape(X)[0] # The number of frames in the TDFT
    M = np.shape(X)[1] # The length of each DFT
    x = np.zeros(L * (N - 1) + M, dtype=np.complex128) # The array holding the reconstructed signal

    if truncate == None:
        truncate = M

    for i in tqdm.tqdm(range(N), disable=quiet):
        # Use the overlap-add method: as long as the shifted windowing functions (times f) add up to a constant 1, we simply
        # IFFT each frame, multiply by f, and add it to where it was taken from
        x[L * i:L * i + truncate] += f * ifft(X[i, :])[:truncate]

    return x

