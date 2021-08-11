"""
This code is from the ants package: https://github.com/lermert/ants_2.

:copyright:
    noisi development team
:license:
    GNU Lesser General Public License, Version 3 and later
    (https://www.gnu.org/copyleft/lesser.html)

"""

import numpy as np
from obspy.signal.filter import envelope
from obspy.signal.util import next_pow_2
from scipy import fftpack
from scipy.signal import iirfilter, zpk2sos
from noisi.ants.tools.windows import my_centered

def bandpass(freqmin, freqmax, df, corners=4):
    """
    From obspy with modification.

    Butterworth-Bandpass Filter.

    Filter data from ``freqmin`` to ``freqmax`` using ``corners``
    corners.
    The filter uses :func:`scipy.signal.iirfilter` (for design)
    and :func:`scipy.signal.sosfilt` (for applying the filter).

    :type data: numpy.ndarray
    :param data: Data to filter.
    :param freqmin: Pass band low corner frequency.
    :param freqmax: Pass band high corner frequency.
    :param df: Sampling rate in Hz.
    :param corners: Filter corners / order.
    :param zerophase: If True, apply filter once forwards and once backwards.
        This results in twice the filter order but zero phase shift in
        the resulting filtered trace.
    :return: Filtered data.
    """
    fe = 0.5 * df
    low = freqmin / fe
    high = freqmax / fe
    # raise for some bad scenarios
    if high > 1:
        high = 1.0
        msg = "Selected high corner frequency is above Nyquist. " + \
              "Setting Nyquist as high corner."
        warnings.warn(msg)
    if low > 1:
        msg = "Selected low corner frequency is above Nyquist."
        raise ValueError(msg)
    z, p, k = iirfilter(corners, [low, high], btype='band',
                        ftype='butter', output='zpk')
    sos = zpk2sos(z, p, k)
    return sos


def whiten_taper(ind_fw1,ind_fw2,npts,taper_samples):
    
    if ind_fw1 - taper_samples >= 0:
        i_l = ind_fw1 - taper_samples
    else:
        i_l = 0
        print('** Could not fully taper during whitening. Consider using a \
            smaller frequency range for whitening.')

    if ind_fw2 + taper_samples < npts:
        i_h = ind_fw2 + taper_samples
    else:
        i_h = npts - 1
        print('** Could not fully taper during whitening. Consider using a \
            smaller frequency range for whitening.')
    
    
    taper_left = np.linspace(0.,np.pi/2,ind_fw1-i_l)
    taper_left = np.square(np.sin(taper_left))
    
    taper_right = np.linspace(np.pi/2,np.pi,i_h-ind_fw2)
    taper_right = np.square(np.sin(taper_right))
    
    taper = np.zeros(npts)
    taper[ind_fw1:ind_fw2] += 1.
    taper[i_l:ind_fw1] = taper_left
    taper[ind_fw2:i_h] = taper_right

    return taper


def whiten_trace(tr,freq1,freq2,taper_samples):

    # zeropadding should make things faster
    n_pad = next_pow_2(tr.stats.npts)

    data = my_centered(tr.data,n_pad)

    freqaxis=np.fft.rfftfreq(tr.stats.npts,tr.stats.delta)

    ind_fw = np.where( ( freqaxis > freq1 ) & ( freqaxis < freq2 ) )[0]

    if len(ind_fw) == 0:
        return(np.zeros(tr.stats.npts))

    ind_fw1 = ind_fw[0]
    
    ind_fw2 = ind_fw[-1]
    
    # Build a cosine taper for the frequency domain
    #df = 1/(tr.stats.npts*tr.stats.delta)
    
    # Taper 
    white_tape = whiten_taper(ind_fw1,ind_fw2,len(freqaxis),taper_samples)
    
    # Transform data to frequency domain
    tr.taper(max_percentage=0.05, type='cosine')
    spec = np.fft.rfft(tr.data)
    
    # Don't divide by 0
    #tol = np.max(np.abs(spec)) / 1e5
    #spec /= np.abs(spec+tol)
    
    # whiten. This elegant solution is from MSNoise:
    spec =  white_tape * np.exp(1j * np.angle(spec))
    
    # Go back to time domain
    # Difficulty here: The time fdomain signal might no longer be real.
    # Hence, irfft cannot be used.
    spec_neg = np.conjugate(spec)[::-1]
    spec = np.concatenate((spec,spec_neg[:-1]))

    tr.data = np.real(np.fft.ifft(spec))


def whiten(tr, freq1, freq2, taper_samples):
    try:
        whiten_trace(tr, freq1, freq2, taper_samples)
    except AttributeError:
        for t in tr:
            whiten_trace(t, freq1, freq2, taper_samples)
    
    
def cap_trace(tr,cap_thresh):
    
    std = np.std(tr.data*1.e6)
    gllow = cap_thresh * std * -1
    glupp = cap_thresh * std
    tr.data = np.clip(tr.data*1.e6,gllow,glupp)/1.e6

def cap(tr, cap_thresh):
    try:
        cap_trace(tr, cap_thresh)
    except AttributeError:
        for t in tr:
            cap_trace(t, cap_thresh)
    #return tr
    
def ram_norm_trace(tr,winlen,prefilt=None):
    
    trace_orig = tr.copy()
    hlen = int(winlen*tr.stats.sampling_rate/2.)

    if 2*hlen >= tr.stats.npts:
        tr.data = np.zeros(tr.stats.npts)
        return()


    weighttrace = np.zeros(tr.stats.npts)
    
    if prefilt is not None:
        tr.filter('bandpass',freqmin=prefilt[0],freqmax=prefilt[1],\
        corners=prefilt[2],zerophase=True)
        
    envlp = envelope(tr.data)

    for n in range(hlen,tr.stats.npts-hlen):
        weighttrace[n] = np.sum(envlp[n-hlen:n+hlen+1]/(2.*hlen+1))
        
    weighttrace[0:hlen] = weighttrace[hlen]
    weighttrace[-hlen:] = weighttrace[-hlen-1]
    
    tr.data = trace_orig.data / weighttrace
   
def ram_norm(tr,winlen,prefilt=None):
    try:
        ram_norm_trace(tr, winlen,prefilt=None)
    except AttributeError:
        for t in tr:
            ram_norm_trace(t,winlen,prefilt=None)