from __future__ import print_function
import numpy as np
from obspy import Stream, Trace
from scipy.signal import hann, tukey
from math import ceil
from sys import float_info

import matplotlib.pyplot as plt

def merge_traces(data, Fs, n_interp=0,maxgap=10.0, ofid=None):
    
    """
    Small script to merge traces over short gaps. Gaps are filled with zeroes.
    Intended to close short gaps of a couple of seconds.
    
    n_interp: Number of samples for interpolation within gaps
    data: obspy stream, where all the traces you would like to merge are collected already
    Fs: List of original sampling rates; it is checked whether the traces have some weird deviating sampling rate
    maxgap: float, maximum length of gap to be filled with zeros. Default is 10 seconds.
    Every gap longer than this value will remain a gap.
    
    
    """

    # check sampling rates and dtypes
    
    data.sort(keys=['network', 'station', 'location', 'channel', 'starttime', 'endtime'])

    # build up dictionary with lists of traces with same ids
    traces_dict = {}
    # using pop() and try-except saves memory
    try:
        while True:
            
            trace = data.pop(0)
            # Skip empty traces (if any)
            if trace.stats.npts <= 0:
                continue

            trace.stats.sampling_rate = \
            round(trace.stats.sampling_rate, 4)
            # Throw data with the wrong sampling rate out.
            if trace.stats.sampling_rate not in Fs:
                print('Bad sampling rate: %g on trace %s' 
                %(trace.stats.sampling_rate,trace.id), file=ofid)
                continue

            # add trace to respective list or create that list
            traces_dict.setdefault(trace.id, []).append(trace)
            
            
    except IndexError:
        pass

    # 'data' contains no traces now; fill it with merged content
    
    # loop through ids
    for id in list(traces_dict.keys()):
        
        trace_list = traces_dict[id]
        cur_trace = Stream(trace_list.pop(0))

        # work through all traces of same id
        while trace_list:
            trace = trace_list.pop(0)
            
            # Case 1: Overlap 
            # Overlaps should be removed in any case, as we don't want to correlate the data twice; especially if they differ between traces (which is most likely the case due to different tapering etc.)
            # Case 2: Perfectly adjacent 
            # Case 3: Short gap 
            cur_gap = trace.stats.starttime - cur_trace[-1].stats.endtime
            if cur_gap <= maxgap:
                # Case 1, 2, 3: Add to old trace and merge
                cur_trace += trace
                # At the moment, gaps are simply filled with 0. That may cause difficulty for FFTing some traces if they get sharp kinks.
                cur_trace.merge(method=1,
                interpolation_samples=n_interp,
                fill_value=0)
            # Case 4: Long gap 
            else:
                # Case 4: Start a new trace
                # Add to output stream
                data += cur_trace
                # Start new stream
                cur_trace = Stream(trace)
            
        # Add the last trace of this id
        # This is actually a stream, but that is ok, adding streams just adds their traces
        data += cur_trace

    return data

#ToDo: look into this
def trim_next_sec(data,verbose,ofid):
    
    """ 
    Trim data to the full second. Ensures that recordings start and end with the closest sample possible to full second.
    data: Is an obspy stream or trace. The returned stream/trace may be shorter
    ofid: Output file 
    
    """

    

    if isinstance(data,Trace):
        data = Stream(data)
        
    for tr in data:

        starttime=tr.stats.starttime
        sec_to_remove=tr.stats.starttime.microsecond/1e6
        sec_to_add=1.-sec_to_remove

        if sec_to_add > tr.stats.delta:
            if verbose: 
                print('* Trimming to full second.\n',file=ofid)
            tr.trim(starttime=tr.stats.starttime+sec_to_add,
                nearest_sample=True)
            
    return data

def slice_traces(data, len_sec, min_len_sec, verbose, ofid):

    """
    Slice an ObsPy stream object with multiple traces; The stream of new 
    (sliced) traces merely contains references to the original trace.
    """
    
    s_new=Stream()
    
    #- set initial start time
    data.sort(['starttime'])
    
    
    
    for k in np.arange(len(data)):
        
        starttime=data[k].stats.starttime
        #- march through the trace until the endtime is reached
        while starttime < data[k].stats.endtime-min_len_sec:
            
            s_new += data[k].slice(starttime,starttime + len_sec-
            data[k].stats.delta)
            
            starttime += len_sec
            
    n_traces=len(s_new)
    
    if verbose:
        print('* contains %g trace(s)' %n_traces,file=ofid)
        
    return s_new
    



def event_exclude(trace,windows,n_compare,freq_min,freq_max,\
factor_enrg=1.,taper_perc=0.05,thresh_stdv=1.,ofid=None,verbose=False):

    """
    A really somewhat complicated way to try and get rid of earthquakes and other high-energy bursts. 
    A sort of coarse multiwindow-trigger; I haven't found a better (comparatively fast) way so far.
    High-energy events will be replaced by zero with their sides tapered.
    Operates directly on the trace.
    """
    if trace.stats.npts == 0:
        return()

    weight = np.ones(trace.stats.npts)
    windows.sort() # make sure ascending order

    testtrace = trace.copy()
    testtrace.taper(type='cosine',max_percentage = taper_perc)
    testtrace.filter('bandpass',freqmin = freq_min,freqmax = freq_max,
                      corners = 3, zerophase = True)
    # length of taper dep on minimum frequency that 
    # should be available

    n_hann = int(trace.stats.sampling_rate / freq_min)
    tpr = hann(2*n_hann)
    
    for win in windows: 
    # initialize arrays of subtrace values (energy, standard deviation) for each window length; maybe just use an expandable list (max. a couple of hundred values)
        enrg = []
        stdv = []
        t = []
        marker = []
        weighting_trace = np.ones(trace.stats.npts)
    # fill those arrays
        t0 = trace.stats.starttime
        while t0 < trace.stats.endtime-win:
            
            subtr = testtrace.slice(starttime=t0,endtime=t0+win-1).data

            enrg.append(np.sum(np.power(subtr,2))/win)
            subwin = int(win/3)
            [a,b,c] = [ np.std(subtr[0:subwin]),
                        np.std(subtr[subwin:2*subwin]),
                        np.std(subtr[2*subwin:3*subwin]) ]  
                              
               
            stdv.append(np.max([a,b,c])/(np.min([a,b,c])+float_info.epsilon))
            t.append((t0+win/2).strftime('%s'))
            t0 += win
        
        # count how many windows are excluded on counter
        # step through the array enrg, stdv; this should be relatively fast as array are not particularly long


        winsmp = int(ceil(win*trace.stats.sampling_rate))
        #extsmp = int(0.75 * winsmp)
         
        for i in range(2,len(enrg)):

            sc = int((i+0.5) * winsmp)

            i0 = i - n_compare if i>=n_compare else 0
            i1 = i0 + n_compare 
            if i1 >= len(enrg):
                i0 = i0 - (len(enrg)-i1)
                i1 = len(enrg)
                
            mean_enrg = np.mean(enrg[i0:i1])
            
            if enrg[i] > factor_enrg * mean_enrg and stdv[i] > thresh_stdv:
                
                j0 = sc - int(0.5*winsmp)#extsmp
                j1 = sc + int(0.5*winsmp)#extsmp


                marker.append(1)
                weighting_trace[j0:j1] *= 0.
                #weighting_trace[i*winsmp:(i+1)*winsmp] *= 0.
                #if i*winsmp > n_hann:
                if j0 > n_hann:
                    weighting_trace[j0-n_hann:j0] *= 1-tpr[0:n_hann]
                    #weighting_trace[i*winsmp-n_hann:i*winsmp] *= 1-tpr[0:n_hann]
                else:
                    weighting_trace[0:j0] *= 1-tpr[n_hann-j0:n_hann]
                    #weighting_trace[0:i*winsmp] *= 1-tpr[n_hann-i*winsmp:n_hann]
                
                #if (i+1)*winsmp+n_hann <=len(weighting_trace):
                if j1 + n_hann <= len(weighting_trace):
                    weighting_trace[j1:j1+n_hann] *= 1-tpr[n_hann:]
                    #weighting_trace[(i+1)*winsmp:(i+1)*winsmp+n_hann] *= 1-tpr[n_hann:]
                else:
                    weighting_trace[j1:] *= 1-tpr[n_hann:n_hann+len(weighting_trace)-j1]
                    #weighting_trace[(i+1)*winsmp:] *= 1-tpr[n_hann:n_hann+len(weighting_trace)-(i+1)*winsmp]
                # build in that if taper is longer than trace itself, it gets shortened.
            else:
                marker.append(0)
        #plt.plot(trace.times()+int(trace.stats.starttime.strftime('%s')),weighting_trace)
        weight *= weighting_trace        
        marker=np.array(marker)
        enrg=np.array(enrg)
        

    trace.data *= weight
    # percentage of data that was cut:
    pctg_cut=float(np.sum(weight==0.))/float(trace.stats.npts)*100
    # display a summary of how much was kept 
    if verbose and pctg_cut > 0:
        print('cut %g percent of data from trace: ' 
            %pctg_cut,file=ofid)
        print(trace.id,file=ofid)
    

    return()
 
def get_event_filter(catalogue, Fs, t0, t1):

    """
    Create a time-domain filter removing all events with Mw > 5.6
    according to GCMT catalogue and
    the empirical rule of Ekstroem (2001):
    T = 2.5 + 40*(Mw-5.6) [hours]
    catalogue: obspy catalogue object
  """
    event_filter_list = []
    # nsamples = int((t1-t0) * Fs)
    # event_filter = Trace(data=np.ones(nsamples))
    # event_filter.stats.starttime = t0
    # event_filter.stats.sampling_rate = Fs

    for cat in catalogue[::-1]:
        # get origin time
        t_o = cat.origins[0].time
        print('Original catalog onset: ' + t_o.strftime("%Y.%j.%H:%M:%S"))
        t_start = t_o - 10

        if len(event_filter_list) > 0:
            if t_o < event_filter_list[-1][1]:
                t_start = event_filter_list[-1][0]
                event_filter_list.pop()

        print('Selected onset: '+t_start.strftime("%Y.%j.%H:%M:%S"))
        # get magnitude
        m = cat.magnitudes[0].mag
        print("Magnitude ", m)
        if cat.magnitudes[0].magnitude_type != 'MW':
            raise ValueError('Magnitude must be moment magnitude.')
        if m < 5.6:
            print('Event Mw < 5.6: Not ok with Ekstroem event exclusion rule.\
 Skipping event.')
            continue

        # determine T
        T = 2.5 * 3600 + 61.8 * (m-5.6) * 3600 

        event_filter_list.append([t_start,t_start+T+10])

        # Add some time for taper!
        #n_taper = int(T * Fs * 0.05)
        #n_seg = int(T * Fs + n_taper)
        
       

        # mute T - tapered window
        #seg = np.ones(n_seg)# create a window tapering down to zero in the middle.
        
        #tap = tukey(n_seg,0.05)
        #seg *= tap
        #seg *= -1.
        #seg += 1.

        # sample where it starts
        #i_seg = int((t_o - t0) * Fs) - n_taper
        
       
        #if i_seg >= 0 and i_seg+(T*Fs)+n_taper < len(event_filter.data):
        #    event_filter.data[i_seg:i_seg+int(T*Fs)+n_taper] *= seg
        #elif i_seg < 0 and i_seg+(T*Fs)+n_taper < len(event_filter.data):
        #    event_filter.data[0:i_seg+int(T*Fs)+n_taper] *= seg[-i_seg:]
        #elif i_seg >=0 and i_seg+(T*Fs)+n_taper > len(event_filter.data):
        #    event_filter.data[i_seg:] *= seg[:(len(event_filter.data)-(i_seg+int(T*Fs)+n_taper))]
        #else:
        #    raise ValueError('Something went wrong: check config parameters exclude_start,\
        #        exclude_end')


    return event_filter_list





