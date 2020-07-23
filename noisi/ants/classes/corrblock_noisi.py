# Correlation block object:
#from __future__ import print_function
from obspy import Stream, UTCDateTime, read
from scipy.signal import sosfilt
import numpy as np
import re

from noisi.ants.tools.util_noisi import get_geoinf
from noisi.ants.classes.corrtrace_noisi import CorrTrace
from noisi.ants.tools.correlations import cross_covar, interference, pcc_2
from noisi.ants.tools.treatment import ram_norm, whiten, cap, bandpass
# list of possible channels combinations
horizontals = ['RR', 'RT', 'TR', 'TT', 'TZ', 'ZT', 'RZ', 'ZR']
import os #, psutil
#from pympler import muppy, summary, tracker


class CorrBlock(object):

    # - initialize with station pairs
    def __init__(self, block, cfg):

        self.cfg = cfg
        self._correlations = {}
        self.inv = block.inventory
        self.channels = block.channels
        self.station_pairs = block.station_pairs
        self.channel_pairs = block.channel_pairs
        t0 = UTCDateTime(self.cfg.time_begin)
        self.t0 = self.initialize_data(t0)
        self.sampling_rate = self.data[0].stats.sampling_rate
        self.delta = self.data[0].stats.delta
        if self.cfg.bandpass is not None:
            fmin = self.cfg.bandpass[0]
            fmax = self.cfg.bandpass[1]
            if fmax > 0.5 * self.sampling_rate:
                raise ValueError("Upper filter freq above Nyquist.")
            if fmax <= fmin:

                msg = "Bandpass upper corner frequency\
must be above lower corner frequency."
                raise ValueError(msg)

            order = self.cfg.bandpass[2]
            self.sos = bandpass(freqmin=fmin, freqmax=fmax,
                                df=self.sampling_rate, corners=order)

        for cp in self.channel_pairs:
            for pair in cp:
                if self.cfg.rotate:
                    pair = [re.sub('E$', 'T', str) for str in pair]
                    pair = [re.sub('N$', 'R', str) for str in pair]
                cp_name = '{}--{}'.format(*pair)
                prepstring = self.get_prepstring()

                self._correlations[cp_name] =\
                    CorrTrace(cfg,pair[0], pair[1], self.sampling_rate,
                              stck_int=cfg.interm_stack, prepstring=prepstring,
                              window_length=cfg.time_window_length,
                              corr_type=cfg.corr_type, maxlag=cfg.corr_maxlag,
                              t0=self.t0,
                              t1=UTCDateTime(self.cfg.time_end),
                              overlap=cfg.time_overlap, corr_params=None,
                              rotate=cfg.rotate)

        if any([i in cfg.corr_tensorcomponents for i in horizontals]):
            self.baz1 = []
            self.baz2 = []
            for pair in self.channel_pairs[0]:
                #print(pair)
                try:
                    geoinf = get_geoinf(cfg.project_path,pair[0], pair[1])
                    # find azimuth, backazimuth
                    self.baz2.append(geoinf[5])
                    self.baz1.append(geoinf[6])
                except FileNotFoundError:
                    self.baz2.append(0)
                    self.baz1.append(0)


    def run(self, output_file=None):
        #process = psutil.Process(os.getpid())
        #print('Working on station pairs:')
        #for sta in self.station_pairs:
        #    print("{}--{}".format(sta[0], sta[1]))
        t_0 = self.t0
        t_end = UTCDateTime(self.cfg.time_end)
        win_len_seconds = self.cfg.time_window_length
        min_len_samples = int(round(self.cfg.time_min_window *
                                    self.sampling_rate))
        max_lag_samples = int(round(self.cfg.corr_maxlag * self.sampling_rate))



        # Time loop
        t = t_0
        t_old = t_0
        # mytracker = tracker.SummaryTracker()
        while t < t_end:
            print(t, file=output_file, end="\n")
            #print("Memory usage in Gb loop begin ", process.memory_info().rss / 1.e9, 
            #      file=output_file, end="\n")
            
            if self.channels == []:
                break
            if len(self.data) == 0:
                break

            #for corr_k, corr_v in self._correlations.items():
            #    summary.print_(summary.summarize(corr_v))
            # if "windows" in locals():
            #     del windows
            
            windows = self.data.slide(win_len_seconds - self.delta, win_len_seconds - self.cfg.time_overlap,
                                      offset=(t - self.data[0].stats.starttime),
                                      include_partial_windows=False)

            for w in windows:
                

                t += self.cfg.time_window_length - self.cfg.time_overlap
                if True in [wt.stats.endtime - wt.stats.starttime < (win_len_seconds - self.delta)\
                            for wt in w]:
                    # skip this window
                    continue


                if w[0].stats.endtime > t_end:
                    break

                # Apply preprocessing
                w = self.preprocess(w)
                # may return a deepcopy if non-linear processing is applied.
                # - station pair loop
                    

                
                for sp_i, pair in enumerate(self.station_pairs):

                    # - select traces
                    [net1, sta1] = pair[0].split('.')
                    [net2, sta2] = pair[1].split('.')
                    str1 = w.select(network=net1, station=sta1)
                    str2 = w.select(network=net2, station=sta2)

                    # - if horizontal components are involved, copy and rotate
                    if any([i in self.cfg.corr_tensorcomponents
                            for i in horizontals]) and self.cfg.rotate:
                        str1, str2 = self.rotate(str1, str2,
                                                 self.baz1[sp_i],
                                                 self.baz2[sp_i])
                    for cpair in self.channel_pairs[sp_i]:

                        if self.cfg.rotate:
                            cpair = [re.sub('E$', 'T', str) for str in cpair]
                            cpair = [re.sub('N$', 'R', str) for str in cpair]

                        cp_name = '{}--{}'.format(*cpair)

                        loc1, cha1 = cpair[0].split('.')[2:4]
                        loc2, cha2 = cpair[1].split('.')[2:4]
                        
                        #print('str1', str1)
                        #print('cha1', cha1)
                        #print('str2', str2)
                        #print('cha2', cha2)

                        try:
                            tr1 = str1.select(location=loc1, channel=cha1)[0]
                            tr2 = str2.select(location=loc2, channel=cha2)[0]
                        except IndexError:
                            print("Channel not found", file=output_file)
                            continue

                        # - check minimum length requirement
                        # - Quite often not fulfilled due to data gaps
                        traces_ok = self.perform_checks(tr1, tr2, output_file,
                                                        min_len_samples)
                        if not traces_ok:
                            continue

                        if self.cfg.corr_type == 'ccc':
                            # correlate
                            correlation = cross_covar(tr1.data, tr2.data,
                                                      max_lag_samples,
                                                      self.cfg.corr_normalize)[0]
                        elif self.cfg.corr_type == 'mic':
                            correlation = interference(tr1.data, tr2.data,
                                                       max_lag_samples)
                        elif self.cfg.corr_type == 'pcc':
                            correlation = pcc_2(tr1.data, tr2.data,
                                                max_lag_samples)[0]

                        # add to stack
                        if len(correlation) == 2 * max_lag_samples + 1:
                            self._correlations[cp_name]._add_corr(correlation,
                                                                  tr1.stats.starttime)
                            del correlation
                        else:
                            print('Empty window.',
                                  file=output_file)
                #t += self.cfg.time_window_length - self.cfg.time_overlap
            if t == t_old:
                t += self.cfg.time_window_length - self.cfg.time_overlap
                
                
                
            #print("Trying update at time ", t)

            self.update_data(t)
            
            if len(self.data) == 0:
                break

            # check if there is a gap
            while t < self.data[0].stats.starttime - self.cfg.time_overlap:
                t += win_len_seconds - self.cfg.time_overlap
                #print("jumping to t ", t)
            t_old = t
            
            # summary.print_(summary.summarize(self.data))
        
        # - Write results
        for corr in self._correlations.values():
            corr.write_stack(self.cfg,output_format=self.cfg.format_output)

        #print('Finished a correlation block.')

    def perform_checks(self, tr1, tr2, output_file, min_len_samples):
        #if tr1.stats.starttime != tr2.stats.starttime:
        #    print("Traces are not synchronous.", file=output_file)
        #    return(False)

        if tr1.stats.npts < min_len_samples:
            print("Trace length < min samples\n", file=output_file)
            return(False)

        if tr2.stats.npts < min_len_samples:
            print("Trace length < min samples\n", file=output_file)
            return(False)

        if True in np.isnan(tr1.data):
            print("Trace contains nan\n", file=output_file)
            return(False)

        if True in np.isnan(tr2.data):
            print("Trace contains nan\n", file=output_file)
            return(False)

        if True in np.isinf(tr1.data):
            print("Trace contains inf\n", file=output_file)
            return(False)

        if True in np.isinf(tr2.data):
            print("Trace contains inf\n", file=output_file)
            return(False)

        return(True)

    # debugging @profile
    def preprocess(self, tr):

        if True in [self.cfg.cap_glitch, 
                    self.cfg.whiten,
                    self.cfg.onebit,
                    self.cfg.ram_norm] or\
            self.cfg.time_overlap != 0.0:
            tr_temp = Stream()
            for t in tr:
                tr_temp += t.copy()
            tr = tr_temp

        # linear
        if self.cfg.bandpass is not None:
            w_temp = sosfilt(self.sos, tr.data)
            tr.data = sosfilt(self.sos, w_temp[::-1])[::-1]

        # non-linear
        if self.cfg.cap_glitch:
            cap(tr, self.cfg.cap_thresh)

        if self.cfg.whiten:
            whiten(tr, self.cfg.white_freqmin,
                   self.cfg.white_freqmax,
                   self.cfg.white_taper_samples)

        if self.cfg.onebit:
            for t in tr:
                t.data = np.sign(t.data)
                
        if self.cfg.ram_norm:
            ram_norm(tr, self.cfg.ram_window, self.cfg.ram_prefilt)

        return(tr)

    # debugging @profile
    def rotate(self, str1, str2, baz1, baz2):
        s_temp1 = str1.copy()
        s_temp2 = str2.copy()

        try:
            s_temp1.rotate('NE->RT', back_azimuth=baz1)
        except:
            if str1[0].stats.station != str2[0].stats.station:
                print('** data not rotated for stream: ')
                print(s_temp1)
            pass

        try:
            s_temp2.rotate('NE->RT', back_azimuth=baz2)
        except:
            if str1[0].stats.station != str2[1].stats.station:
                print('** data not rotated for stream: ')
                print(s_temp2)
            pass
        del str1, str2
        return(s_temp1, s_temp2)

    # debugging @profile
    def update_data(self, t):
        # mytracker = tracker.SummaryTracker()
        # mytracker.print_diff()
        # add a new round of data:
        for ix_c, channel in enumerate(self.channels):

            if len(self.data.select(id=channel)) > 0:
                if self.data.select(id=channel)[-1].stats.endtime > t + self.cfg.time_window_length:
                    continue

            while True:
                try:
                    f = self.inv[channel].pop(0)
                except IndexError:
                    # No more data. Remove this channel
                    self.channels.pop(ix_c)
                    break
                try:
                    self.data += read(f)
                    #print("read trace to ", self.data[-1].stats.endtime)
                    # success
                    break
                except IOError:
                    #print("** Could not read trace: %s" % f)
                    pass
                    # try the next file
        self.data._cleanup()
        self.data.sort(keys=["starttime"])
        self.data.trim(starttime=t)
        # mytracker.print_diff()
        return()

    def initialize_data(self, t0):
        # t0: begin time of observation
        t_min = t0 + self.cfg.time_window_length - self.cfg.time_overlap
        self.data = Stream()

        for channel in self.channels:
            ttemp = 0
            while ttemp < t_min:
                f = self.inv[channel].pop(0)
                try:
                    self.data += read(f)
                    ttemp = self.data[-1].stats.endtime
                except IOError:
                    print('** problems reading file %s'
                          % self.inv.data[channel])

        earliest_start_date = min([tr.stats.starttime for tr in self.data])
        self.data.trim(max(t0, earliest_start_date))
        self.data._cleanup()
        return(max(t0, earliest_start_date))

    def get_prepstring(self):

        prepstring = ''
        if self.cfg.bandpass:
            prepstring += 'b'
        else:
            prepstring += '-'
        if self.cfg.cap_glitch:
            prepstring += 'g'
        else:
            prepstring += '-'
        if self.cfg.whiten:
            prepstring += 'w'
        else:
            prepstring += '-'
        if self.cfg.onebit:
            prepstring += 'o'
        else:
            prepstring += '-'
        if self.cfg.ram_norm:
            prepstring += 'r'
        else:
            prepstring += '-'

        return prepstring
