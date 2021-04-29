import numpy as np
from numba import njit
import tqdm

from dsplib.utils import *

@njit
def dwt(x, scaling_coeffs):
    '''
    Given a signal and the scaling coefficients of the wavelet, computes one level
    of the discrete time wavelet transform.

    Parameters
    ----------
    x : float[N]
        An N-point signal. If N is not a power of 2, the signal will be zero padded.

    scaling_coeffs : float[M]
        The M-point array of scaling coefficients that determine the wavelet.

    Returns
    -------
    coeff_levels : [float[L], float[L]]
        An array containing the approximation and detail coefficients, respectively.
    '''

    # Get the length of the scaling coefficients. These begin their indexing at -1, not 0!
    N = len(scaling_coeffs)

    # The low pass tap weights are the scaling coefficients in reverse order.
    g = scaling_coeffs[::-1]

    # The high pass tap weights are the scaling coefficients with every odd index negated.
    # This is a consequence of the QMF pair h and g make.
    h = scaling_coeffs * (-1)**(np.arange(N) + 1)

    # Compute the approximation and detail coefficients by filtering the signal with g
    # and h before downsampling by a factor of 2. Since g and h start at -1 and we want the
    # portion of the convolution that starts at 0, we must take [1:]
    approx_coeffs = downsample(convolve(x, g)[1:])
    detail_coeffs = downsample(convolve(x, h)[1:])

    # Return the computed wavelet transform coefficients
    return [approx_coeffs, detail_coeffs]


def idwt(approx_coeffs, detail_coeffs, scaling_coeffs):
    '''
    Given the approximation and detail coefficients, computes the original signal.

    Parameters
    ----------
    approx_coeffs : float[N]
        An array of the approximation coefficients.

    detail_coeffs : float[N]
        An array of the detail coefficients.

    scaling_coeffs : float[M]
        The M-point array of scaling coefficients that determine the wavelet.

    Returns
    -------
    x : float[L]
        The reconstructed signal.
    '''

    # Get the length of the scaling coefficients. These begin their indexing at -1, not 0!
    N = len(scaling_coeffs)

    # The low pass tap weights are the scaling coefficients in reverse order.
    g = scaling_coeffs[::-1]

    # The high pass tap weights are the scaling coefficients with every odd index negated.
    # This is a consequence of the QMF pair h and g make.
    h = scaling_coeffs * (-1)**(np.arange(N) + 1)

    # The reconstruction filters are the time-reverse of the analysis filters.
    gr = g[::-1]
    hr = h[::-1]

    # Compute the upsampled, reconstruction coefficients by filtering the upsampled signal with gr
    # and hr. Since g and h start at -1 and we want the portion of the convolution that starts at N - 2
    # (since the filters are time reversed). There also seems to be an extra zero, so we take [N-2:-1]
    upsampled_approx = convolve(upsample(approx_coeffs), gr)[N - 2:]
    upsampled_detail = convolve(upsample(detail_coeffs), hr)[N - 2:]

    # Return the sum of the reconstruction coefficients: the reconstructed signal.
    return upsampled_approx + upsampled_detail


def multidwt(x, scaling_coeffs, levels=None):
    '''
    Given a signal and the scaling coefficients of the wavelet, computes multiple
    levels of the discrete time wavelet transform.

    Parameters
    ----------
    x : float[N]
        An N-point signal. If N is not a power of 2, the signal will be zero padded.

    scaling_coeffs : float[M]
        The M-point array of scaling coefficients that determine the wavelet.

    levels : int (optional)
        The number of levels of the wavelet transform to compute. Defaults to None, which results
        in the maximum number of non-redundant levels.

    Returns
    -------
    coeff_levels : [float[L0], float[L0], float[L1], ..., float[LK]]
        An array containing the bottommost approximation and detail coefficients,
        as well as the other layers of detail coefficients.
    '''

    # Zero pad the signal to a power of 2
    Nx = len(x)
    Nx_log_2 = np.ceil(np.log2(Nx)).astype(int)
    x_v = extend_array(x)

    # If the user does not specify the number of levels, take it to be the maximum number
    # Since we downsample the signal by 2 until we hit a 1-point signal, the maximum
    # number of non-redundant levels is log2(N), where N is the signal length
    if levels == None:
        levels = Nx_log_2

    # If the user specifies a number of levels greater than log2(N), we throw an error
    # The algorithms user here cannot reconstruct a signal from such an array of coefficients
    if levels > Nx_log_2:
        raise ValueError('Too many levels. Signal not recoverable using built algorithms.')

    # Recursively compute the levels of approximation and detail coefficients
    # Every level's set of coefficients is computed from the previous level's
    # approximation coefficients.
    coeff_levels = []
    approx_coeffs = np.copy(x)
    for level in range(levels):
        approx_coeffs, detail_coeffs = dwt(approx_coeffs, scaling_coeffs)
        coeff_levels.insert(0, detail_coeffs)

    # Don't forget to add the approximation coefficients of the last level!
    coeff_levels.insert(0, approx_coeffs)

    # Return the list of the various coefficient levels
    return coeff_levels


def multiidwt(coeff_levels, scaling_coeffs):
    '''
    Given an array of coefficient levels and the scaling coefficients of the wavelet,
    reconstructs the original signal.

    Parameters
    ----------
    coeff_levels : [float[L0], float[L0], float[L1], ..., float[LK]]
        An array containing the bottommost approximation and detail coefficients,
        as well as the other layers of detail coefficients.

    scaling_coeffs : float[M]
        The M-point array of scaling coefficients that determine the wavelet.

    Returns
    -------
    x : float[L]
        The reconstructed signal.
    '''

    # The number of coefficient levels
    levels = len(coeff_levels)

    # Loops through each level, replacing the next one by the IDWT of the previous one (the approximation
    # coefficients) and the one we're replacing (the detail coefficients). With non-Haar wavelets, we sometimes
    # generate too many zeros, so we truncate each reconstructed signal to be no bigger than the next level of
    # coefficients.
    coeff_levels_copy = coeff_levels.copy()
    for i in range(levels - 1):
        coeff_levels_copy[i + 1] = idwt(coeff_levels_copy[i], coeff_levels_copy[i + 1], scaling_coeffs)
        if i < levels - 2:
            coeff_levels_copy[i + 1] = coeff_levels_copy[i + 1][:len(coeff_levels_copy[i + 2])]

    # Return the reconstructed signal, clamped to the nearest lower power of 2 (since the input
    # signal is guaranteed to be a power of 2 and this process has only added entries).
    Nx_log_2 = 2**np.floor(np.log2(len(coeff_levels_copy[-1]))).astype(int)
    return coeff_levels_copy[-1][:Nx_log_2]


def dwt2(x, scaling_coeffs):
    '''
    Calculates the 2D discrete time wavelet transform of a matrix via column-row decomposition.
    We calculate the DTWT of the columns first, then the rows.

    Parameters
    ----------
    x : float[N, N]
        The 2D signal.

    scaling_coeffs : float[M]
        The M-point array of scaling coefficients that determine the wavelet.

    Returns
    -------
    coeff_levels : [float[L], float[L], float[L], float[L]]
        An array containing the AA, AD, DD, and DA coefficients, respectively.
    '''

    # Perform a DWT along the first column of the input matrix.
    a_coeffs, d_coeffs = dwt(x[:, 0], scaling_coeffs)

    # Transpose the resulting approximation and detail coefficients
    # to column vectors.
    a_coeffs = np.transpose(a_coeffs[np.newaxis])
    d_coeffs = np.transpose(d_coeffs[np.newaxis])

    # Repeat the above for the remaining columns.
    for i in range(len(x[0, :]) - 1):
        a_coeffs_n, d_coeffs_n = dwt(x[:, i + 1], scaling_coeffs)
        a_coeffs_n = np.transpose(a_coeffs_n[np.newaxis])
        d_coeffs_n = np.transpose(d_coeffs_n[np.newaxis])

        # Stack the new approximation and detail coefficients horizontally
        # (column wise) to the right of the others.
        a_coeffs = np.hstack((a_coeffs, a_coeffs_n))
        d_coeffs = np.hstack((d_coeffs, d_coeffs_n))

    # Perform DWTs along the first rows of the approx. and detail coefficient
    # matrices
    aa_coeffs, da_coeffs = dwt(a_coeffs[0, :], scaling_coeffs)
    ad_coeffs, dd_coeffs = dwt(d_coeffs[0, :], scaling_coeffs)

    # Repeat the above for the remaining rows.
    for i in range(len(a_coeffs[:, 0]) - 1):
        aa_coeffs_n, da_coeffs_n = dwt(a_coeffs[i + 1, :], scaling_coeffs)
        ad_coeffs_n, dd_coeffs_n = dwt(d_coeffs[i + 1, :], scaling_coeffs)

        # Stack the new approximation and detail coefficients vertically
        # (row wise) below the others.
        aa_coeffs = np.vstack((aa_coeffs, aa_coeffs_n))
        da_coeffs = np.vstack((da_coeffs, da_coeffs_n))
        ad_coeffs = np.vstack((ad_coeffs, ad_coeffs_n))
        dd_coeffs = np.vstack((dd_coeffs, dd_coeffs_n))

    # Return the resulting 4 matrices in a clockwise order starting at the top left
    return [aa_coeffs, ad_coeffs, dd_coeffs, da_coeffs]


def idwt2(aa_coeffs, ad_coeffs, dd_coeffs, da_coeffs, scaling_coeffs):
    '''
    Calculates the inverse 2D discrete time wavelet transform of a matrix via
    row-column decomposition. We calculate the DTWT of the rows first, then the columns.
    (This needs to be opposite the order we perform this in the forward DTWT.)

    Parameters
    ----------
    aa_coeffs : float[N]
        An array of the double approximation coefficients.

    ad_coeffs : float[N]
        An array of the approximation-detail coefficients.

    dd_coeffs : float[N]
        An array of the detail-detail coefficients.

    da_coeffs : float[N]
        An array of the detail-approximation coefficients.

    scaling_coeffs : float[M]
        The M-point array of scaling coefficients that determine the wavelet.

    Returns
    -------
    x : float[L, L]
        The reconstructed signal.
    '''

    a_coeffs = idwt(aa_coeffs[0, :], da_coeffs[0, :], scaling_coeffs)
    d_coeffs = idwt(ad_coeffs[0, :], dd_coeffs[0, :], scaling_coeffs)

    for i in range(len(aa_coeffs[:, 0]) - 1):
        a_coeffs_n = idwt(aa_coeffs[i + 1, :], da_coeffs[i + 1, :], scaling_coeffs)
        d_coeffs_n = idwt(ad_coeffs[i + 1, :], dd_coeffs[i + 1, :], scaling_coeffs)

        a_coeffs = np.vstack((a_coeffs, a_coeffs_n))
        d_coeffs = np.vstack((d_coeffs, d_coeffs_n))

    x = idwt(a_coeffs[:, 0], d_coeffs[:, 0], scaling_coeffs)
    x = np.transpose(x[np.newaxis])

    for i in range(len(a_coeffs[0, :]) - 1):
        x_n = idwt(a_coeffs[:, i + 1], d_coeffs[:, i + 1], scaling_coeffs)
        x_n = np.transpose(x_n[np.newaxis])

        x = np.hstack((x, x_n))

    return x

def multidwt2(x, scaling_coeffs, levels=None):
    '''
    Given a 2D signal and the scaling coefficients of the wavelet, computes multiple
    levels of the 2D discrete time wavelet transform.

    Parameters
    ----------
    x : float[N, N]
        The 2D signal.

    scaling_coeffs : float[M]
        The M-point array of scaling coefficients that determine the wavelet.

    levels : int (optional)
        The number of levels of the wavelet transform to compute. Defaults to None, which results
        in the maximum number of non-redundant levels.

    Returns
    -------
    coeff_levels : [float[L0], float[L0], float[L1], ..., float[LK]]
        An array containing the bottommost AA, AD, DD, and DA coefficients,
        as well as the other layers of detail coefficients.
    '''

    # Zero pad the signal to a power of 2
    Nw = np.shape(x)[0]
    Nh = np.shape(x)[1]
    N_log_2 = np.ceil(np.log2(max(Nh, Nw))).astype(int)
    N_min = 2**N_log_2

    x = np.pad(x, ([0, N_min - Nw], [0, N_min - Nh]))

    # If the user does not specify the number of levels, take it to be the maximum number.
    # Since we downsample the signal by 2 until we hit a 2x2-point signal, the maximum
    # number of non-redundant levels is log2(N) - 1, where N is the signal length.
    if levels == None:
        levels = N_log_2 - 1

    # If the user specifies a number of levels greater than log2(N) - 1, we throw an error.
    # The algorithms user here cannot reconstruct a signal from such an array of coefficients.
    if levels > N_log_2 - 1:
        raise ValueError('Too many levels. Signal not recoverable using built algorithms.')

    # For each level, compute the 2D DWT of the approximation coefficients
    # of the previous level.
    coeff_levels = []
    aa = np.copy(x)
    for level in range(levels):
        aa, ad, dd, da = dwt2(aa, scaling_coeffs)
        coeff_levels.insert(0, da)
        coeff_levels.insert(0, dd)
        coeff_levels.insert(0, ad)

    # Don't forget to add the approximation coefficients of the last level!
    coeff_levels.insert(0, aa)

    return coeff_levels


def multiidwt2(coeff_levels, scaling_coeffs):
    '''
    Given an array of coefficient levels and the scaling coefficients of the wavelet,
    reconstructs the original matrix.

    Parameters
    ----------
    coeff_levels : [float[L0], float[L0], float[L1], ..., float[LK]]
        An array containing the bottommost AA, AD, DD, and DA coefficients,
        as well as the other layers of detail coefficients.

    scaling_coeffs : float[M]
        The M-point array of scaling coefficients that determine the wavelet.

    Returns
    -------
    x : float[L, L]
        The reconstructed signal.
    '''

    # The number of coefficient levels.
    levels = (len(coeff_levels) - 4) // 3 + 1

    # Loop through each level, replacing the set of 4 approx. and detail coefficients by their
    # reconstruction.
    coeff_levels_copy = coeff_levels.copy()
    for i in range(levels):
        new_aa = idwt2(coeff_levels_copy[0], coeff_levels_copy[1], coeff_levels_copy[2], coeff_levels_copy[3], scaling_coeffs)

        # Remove the smallest four DWT matrices and replace them with their reconstruction
        for j in range(4):
            coeff_levels_copy.pop(0)
        coeff_levels_copy.insert(0, new_aa)

        # If we are not at the top level, make sure our reconstructed matrix is the same size
        # as the three matrices it is adjacent to. (This will chop off some zeros.)
        if i < levels - 1:
            Ns = np.shape(coeff_levels_copy[1])
            coeff_levels_copy[0] = coeff_levels_copy[0][:Ns[0], :Ns[1]]

    # Set the reconstructed matrix to be the closest, smaller power of 2 (since we know that
    # it was restricted to this at the input).
    N_log_2 = 2**np.floor(np.log2(np.shape(coeff_levels_copy[0])[0])).astype(int)
    return coeff_levels_copy[0][:N_log_2, :N_log_2]


def unwrapdwt2(coeff_list):
    '''
    Unwraps a coefficient list of a multilevel 2D DWT.

    Parameters
    ----------
    coeff_levels : [float[L0], float[L0], float[L1], ..., float[LK]]
        An array containing the bottommost AA, AD, DD, and DA coefficients,
        as well as the other layers of detail coefficients.

    Returns
    -------
    flattened_coeffs : float[M]
        A single vector of wavelet coefficients.

    shapes : float[K]
        The shapes of the original coefficient levels.
    '''

    flattened_coeffs = []
    shapes = []
    for c in coeff_list:
        flattened_coeffs = np.append(flattened_coeffs, np.copy(c))
        shapes.append(np.shape(c))
    return flattened_coeffs, shapes


def wrapdwt2(flattened_coeffs, shapes):
    '''
    Rewraps a coefficient list of a multilevel 2D DWT.

    Parameters
    -------
    flattened_coeffs : float[M]
        A single vector of wavelet coefficients.

    shapes : float[K]
        The shapes of the original coefficient levels.

    Returns
    ----------
    coeff_levels : [float[L0], float[L0], float[L1], ..., float[LK]]
        An array containing the bottommost AA, AD, DD, and DA coefficients,
        as well as the other layers of detail coefficients.
    '''

    coeff_list = []
    shapes_copy = shapes.copy()
    while len(shapes_copy) > 0:
        s = shapes_copy.pop(0)
        coeff_list.append(np.reshape(np.copy(flattened_coeffs[:s[0]*s[1]]), s))
        flattened_coeffs = flattened_coeffs[s[0]*s[1]:]
    return coeff_list


def displaydwt2(coeff_levels):
    '''
    Stitches together the coefficient levels into a single 2D array so the 2D DWT can
    be displayed as an image.

    Parameters
    ----------
    coeff_levels : [float[L0], float[L0], float[L1], ..., float[LK]]
        An array containing the bottommost AA, AD, DD, and DA coefficients,
        as well as the other layers of detail coefficients.

    Returns
    -------
    A : float[N, N]
        The image of the 2D DWT.
    '''

    levels = (len(coeff_levels) - 4) // 3 + 1
    sizes = []

    coeff_levels_copy = coeff_levels.copy()
    for i in range(levels):
        sizes.append(np.shape(coeff_levels_copy[0]))
        flattened = np.vstack((np.hstack((coeff_levels_copy[0], coeff_levels_copy[1])),np.hstack((coeff_levels_copy[3], coeff_levels_copy[2]))))
        for j in range(4):
            coeff_levels_copy.pop(0)
        if i < levels - 1:
            N = np.shape(coeff_levels_copy[1])
            coeff_levels_copy.insert(0, flattened[:N[0], :N[1]])
        else:
            coeff_levels_copy.insert(0, flattened)

    return coeff_levels_copy[0]
