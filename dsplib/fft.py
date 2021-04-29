import numpy as np
from numba import njit
import tqdm

from dsplib.utils import *

@njit
def bit_reversal_recursive(x):
    '''
    Bit-reverse the addresses of the signal x. This is equivalent to recursively
    sorting the signal into even and odd adresses.

    Parameters
    ----------
    x : float[N]
        The signal to bit reverse.

    Returns
    -------
    y : float[N]
        The bit reversed signal.
    '''

    N = len(x) # The size of the signal

    # If we receive a 2-pt signal, the addresses are already sorted
    if N <= 2:
        return x

    N_over_2 = int(N / 2)

    evens = x[::2] # The even addresses of x
    odds = x[1::2] # The odd addresses of x

    y = np.empty(N, dtype=np.complex128)
    y[0:N_over_2] = bit_reversal_recursive(evens) # Sort the addresses of 'evens' into the even and odd addresses
    y[N_over_2:]  = bit_reversal_recursive(odds) # Sort the addresses of 'odds' into even and odd addresses

    return y


@njit
def twiddle_factors(N):
    '''
    Computes the twiddle factors used in the last stage of butterfly computations.

    Parameters
    ----------
    N : int
        The number of twiddle factors to compute.

    Returns
    -------
    t : float[N]
        Twiddle factors.
    '''

    argu = 2 * np.pi * np.arange(N / 2) / N
    return np.cos(argu) - 1j * np.sin(argu)


@njit
def butterfly_computation(upper, lower, twiddle):
    '''
    Computes the 2-pt DFT (with a twiddle factor) of a 2-pt signal.

    Parameters
    ----------
    upper : float
        The upper value in the block diagram.

    lower : float
        The lower value in the block diagram.

    twiddle : float
        The complex twiddle factor.

    Returns
    -------
    ft_upper : float
        The first complex DFT number.

    ft_lower : float
        The second complex DFT number.
    '''

    modified_lower = twiddle * lower
    return upper + modified_lower, upper - modified_lower


@njit
def fft(x, M):
    '''
    The radix-2 decimation-in-time fast Fourier transform.

    Parameters
    ----------
    x : float[N]
        The signal to Fourier transform.

    M : int
        The length of the FFT.

    Returns
    -------
    X : float[M]
        The M-point FFT of the signal.
    '''

    v = int(np.ceil(np.log2(M)))
    x, V = extend_array(x, v) # We extend the signal to a 2^V signal
    N = 2**V # The exact length of the extended signal
    b = bit_reversal_recursive(x) # We bit-reverse the addresses of the signal
    twiddles = twiddle_factors(N) # and compute the twiddle factors

    # We loop through each butterfly computation stage
    for stage in range(V):
        address_jump = 2**stage # The difference between the upper and lower addresses of butterfly computations within a block
        blocks = 2**(V - stage - 1) # The number of similar butterfly computation blocks in this stage
        block_jump = 2 * address_jump # The difference between the first addresses of each block of butterfly computations
        block_twiddle = twiddles[::2**(V - 1 - stage)] # The twiddle factors for the butterfly computations within this stage

        # We then loop through each block of butterfly computations
        for i in range(blocks):
            # For each block, we perform the same number of butterfly computations as the address jump
            for j in range(address_jump):
                upper_address = i * block_jump + j # The 'upper' address to be used
                lower_address = upper_address + address_jump # The 'lower' address to be used
                b[upper_address], b[lower_address] = butterfly_computation(b[upper_address], b[lower_address], block_twiddle[j]) # Perform the butterfly computation and write the result in place

    return b


@njit
def ifft(x):
    '''
    The IFFT and FFT are very similar operations; we only need to take the complex conjugate of the
    twiddle factors and divide the output by the length of the signal

    Parameters
    ----------
    X : float[N]
        The Fourier transform of the signal.

    Returns
    -------
    x : float[M]
        The recovered signal.
    '''

    X, V = extend_array(X) # We extend the signal to a 2^V signal
    N = 2**V # The exact length of the extended signal
    b = bit_reversal_recursive(X) # We bit-reverse the addresses of the signal
    twiddles = twiddle_factors(N) # and compute the twiddle factors

    # We loop through each butterfly computation stage
    for stage in range(V):
        address_jump = 2**stage # The difference between the upper and lower addresses of butterfly computations within a block
        blocks = 2**(V - stage - 1) # The number of similar butterfly computation blocks in this stage
        block_jump = 2 * address_jump # The difference between the first addresses of each block of butterfly computations
        block_twiddle = np.conj(twiddles[::2**(V - 1 - stage)]) # The twiddle factors for the butterfly computations within this stage (we take the complex conjugate for the IFFT)

        # We then loop through each block of butterfly computations
        for i in range(blocks):
            # For each block, we perform the same number of butterfly computations as the address jump
            for j in range(address_jump):
                upper_address = i * block_jump + j # The 'upper' address to be used
                lower_address = upper_address + address_jump # The 'lower' address to be used
                b[upper_address], b[lower_address] = butterfly_computation(b[upper_address], b[lower_address], block_twiddle[j]) # Perform the butterfly computation and write the result in place

    signal = b / N # We divide the output by the number of samples for the IFFT

    return signal
