import numpy as np
from scipy.signal import hilbert
from math import pi, log
from ants_2.tools.windows import get_window
from ants_2.tools.plot import plot_window     
     

def envelope(correlation,plot=False):
    
    envelope = correlation.data**2 + np.imag(hilbert(correlation.data))**2
    
    return envelope

def windowed_envelope(correlation,plot=False):
    pass    


def windowed_waveform(correlation,g_speed,window_params):
    window = get_window(correlation.stats,g_speed,window_params)
    win = window[0]
    if window[2]:
        win_caus = (correlation.data * win)
        win_acaus = (correlation.data * win[::-1])
        msr = win_caus+win_acaus
    else:
        msr = win-win+np.nan
    return msr


def energy(correlation,g_speed,window_params):
    
    window = get_window(correlation.stats,g_speed,window_params)
    if window_params['causal_side']:
        win = window[0]
    else:
        win = window[0][::-1]
    if window[2]:
        E = np.trapz((correlation.data * win)**2)
        msr = E
        if window_params['plot']:
            plot_window(correlation,win,E)
    else:
        msr = np.nan
        
    return msr
    
def log_en_ratio(correlation,g_speed,window_params):
    delta = correlation.stats.delta
    window = get_window(correlation.stats,g_speed,window_params)
    win = window[0]
    if window[2]:
        E_plus = np.trapz((correlation.data * win)**2) * delta
        E_minus = np.trapz((correlation.data * win[::-1])**2) * delta

        E_ratio = E_plus/(E_minus+np.finfo(E_minus).tiny)

        if E_ratio <= 0:
            msr = np.nan
        else:
            msr = log(E_ratio)
       
        if window_params['plot']:
            wins = win + win[::-1]
            winn = window[1] + window[1][::-1]
            plot_window(correlation,wins,msr,winn)
    else:
        msr = np.nan
    return msr


def get_measure_func(mtype):
    
    if mtype == 'ln_energy_ratio':
        func = log_en_ratio
    elif mtype == 'energy_diff':
        func = energy
    elif mtype == 'square_envelope':
        func = envelope
    elif mtype == 'windowed_waveform_diff':
        func = windowed_waveform
    else:
        msg = 'Measurement %s is not currently implemented.' %mtype
        raise ValueError(msg)
    return func


