  
import numpy as np
from math import sqrt, isnan
from obspy.signal.cross_correlation import xcorr
from scipy.signal import hilbert, correlate
from scipy.fftpack import next_fast_len
import warnings


def my_centered(arr, newsize):
    # get the center portion of a 1-dimensional array
    n = len(arr)
    i0 = (n - newsize) // 2
    if n % 2 == 0:
        i0 += 1
    i1 = i0 + newsize
    return arr[i0:i1]

def running_mean(x, N):
    # adapted from:
    # https://stackoverflow.com/questions/
    # 13728392/moving-average-or-running-mean/27681394#27681394
    ma = np.zeros(len(x), dtype=np.complex)
    cumsum = np.cumsum(np.insert(x, 0, 0))
    ma[: -N + 1] = (cumsum[N:] - cumsum[:-N]) / float(N)
    ma[-N:] = (cumsum[-1] - cumsum[-N]) / float(N)
    return ma


def obspy_xcorr(trace1, trace2, max_lag_samples):

    x_corr = xcorr(trace1.data, trace2.data,
                   max_lag_samples, True)[2]

    return x_corr


def get_correlation_params(data1, data2):

    if len(data1) == 0 or len(data2) == 0:
        return(0, 0, 0, 0, 0, 0)
    # Get the signal energy; most people normalize by the square root of that
    ren1 = np.correlate(data1, data1, mode='valid')[0]
    ren2 = np.correlate(data2, data2, mode='valid')[0]

    # Get the window rms
    rms1 = sqrt(ren1 / len(data1))
    rms2 = sqrt(ren2 / len(data2))

    # A further parameter to 'see' impulsive events:
    # range of standard deviations
    nsmp = int(len(data1) / 4)

    std1 = [0, 0, 0, 0]
    std2 = [0, 0, 0, 0]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(4):

            std1[i] = np.std(data1[i * nsmp: (i + 1) * nsmp])
            if isnan(std1[i]):
                return(0, 0, 0, 0, 0, 0)
            std2[i] = np.std(data2[i * nsmp:(i + 1) * nsmp])
            if isnan(std1[i]):
                return(0, 0, 0, 0, 0, 0)

    # Add small value not to divide by zero
    tol = np.max(std1) * 1e-6
    if tol != 0:
        rng1 = max(std1) / (min(std1) + tol)
        rng2 = max(std2) / (min(std2) + tol)
    else:
        rng1 = 0
        rng2 = 0

    return(rms1, rms2, ren1, ren2, rng1, rng2)


def interference(data1, data2, max_lag_samples):

    if len(data1) == 0 or len(data2) == 0:
        return([], [])
    # a zero mean is assumed.
    data1 -= np.mean(data1)
    data2 -= np.mean(data2)

    interf = np.zeros(2 * max_lag_samples + 1)
    lags = np.arange(-max_lag_samples, max_lag_samples, 1)

    for ix_l in range(len(lags)):
        lag = lags[ix_l]
        if lag < 0:
            interf[ix_l] = np.mean(data1[-lag:] + data2[:lag])
        elif lag == 0:
            interf[ix_l] = np.mean(data1 + data2)
        else:
            interf[ix_l] = np.mean(data1[: -lag] + data2[lag:])

    return(interf)


def cross_covar(data1, data2, max_lag_samples, normalize, params=False):

    if len(data1) == 0 or len(data2) == 0:
        return([], [])

    data1 -= np.mean(data1)
    data2 -= np.mean(data2)

    # Make the data more convenient for C function np.correlate
    data1 = np.ascontiguousarray(data1, np.float32)
    data2 = np.ascontiguousarray(data2, np.float32)

    if params:
        params = get_correlation_params(data1, data2)
        ren1, ren2 = params[2:4]
    else:
        ren1 = np.correlate(data1, data1, mode='valid')[0]
        ren2 = np.correlate(data2, data2, mode='valid')[0]

    if ren1 == 0.0 or ren2 == 0.0 and normalize:
        return([], [])

    # scipy.fftconvolve is way faster than np.correlate
    # and zeropads for non-circular convolution
    ccv = correlate(data2, data1, mode='same')

    if normalize:
        ccv /= (sqrt(ren1) * sqrt(ren2))
    return my_centered(ccv, 2 * max_lag_samples + 1), params


def deconv(data1, data2, max_lag_samples, ma_n=20):

    if len(data1) == 0 or len(data1) == 0:
        return([], [])
    nfft = next_fast_len(2 * len(data1))
    spec1 = np.fft.rfft(data1, n=nfft)
    spec2 = np.fft.rfft(data2, n=nfft)

    smoothed_spectrum2 = running_mean(spec2, N=ma_n)

    crossspec = np.conjugate(spec1) * spec2 / (np.abs(
            np.conjugate(smoothed_spectrum2) * smoothed_spectrum2) +
            np.finfo(spec2.dtype).eps)
    coh = np.fft.irfft(crossspec, n=nfft)
    return(my_centered(coh, 2 * max_lag_samples + 1), [])



def pcc_2(data1, data2, max_lag_samples, params=False):
    # PCC implemented as described by Ventosa et al., SRL 2019
    # doi: 10.1785/0220190022

    data1 = np.ascontiguousarray(data1, np.float64)
    data2 = np.ascontiguousarray(data2, np.float64)

    # determine the analytic signal
    s1 = hilbert(data1)
    s2 = hilbert(data2)

    if params:
        params = get_correlation_params(data1, data2)

    s1 /= (np.abs(s1) + np.finfo(s1.dtype).eps)
    s2 /= (np.abs(s2) + np.finfo(s2.dtype).eps)

    pcc2 = correlate(s2, s1, mode='full', method='fft')
    pcc2 /= len(data1)

    return(my_centered(np.real(pcc2), 2 * max_lag_samples + 1), params)
