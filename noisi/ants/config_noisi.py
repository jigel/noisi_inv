import io
import json
import os
#import click
from obspy import UTCDateTime


DEFAULT_Preprocess = {
    "verbose":True,
    "testrun":False,
    "input_dirs":[],
    "input_format":"MSEED",
    "locations":['','00'],
    "quality_minlengthsec":0.,
    "quality_maxgapsec":0.,
    "gcmt_exclude":False,
    "gcmt_begin":'1970,01,01',
    "gcmt_end":'1970,01,01',
    "gcmt_minmag": 5.6,
    "event_exclude_local_cat":False,
    "event_exclude_local_cat_begin":'1970,01,01',
    "event_exclude_local_cat_end":'1970,01,01',
    "event_exclude_local_cat_minmag":2.0,
    "event_exclude_local_cat_lat":0.00,
    "event_exclude_local_cat_lon":0.00,
    "event_exclude_local_cat_radius":10.0,
    "event_exclude":False,
    "event_exclude_winsec":[],
    "event_exclude_std":2.,
    "event_exclude_n":4,
    "event_exclude_freqmin":0.01,
    "event_exclude_freqmax":1.0,
    "event_exclude_level":2.,
    "wins":True,
    "wins_len_sec":86400,
    "wins_trim":True,
    "wins_detrend":True,
    "wins_demean":True,
    "wins_taper":0.01,
    "wins_taper_type":'cosine',
    "interpolation_samples_gap": 100,
    "Fs_old":[],
    "Fs_new":[],
    "phaseshift": True,
    "Fs_antialias_factor":0.4,
    "instr_correction":True,
    "instr_correction_unit":'VEL',
    "instr_correction_input":'resp',
    "instr_correction_prefilt":[],
    "instr_correction_waterlevel":0.,
}

#CONFIG_Preprocess = os.path.join('input','config_preprocess.json')


class ConfigPreprocess(object):
    """Contains basic parameters for the job (paths, etc.)"""

    def __init__(self,CONFIG_Preprocess):
        
        self.CONFIG_Preprocess = CONFIG_Preprocess
        self.project_path = None
        
        self.verbose = None
        self.testrun = None
        self.input_dirs = None
        self.input_format = None
        self.locations = None

        self.quality_minlengthsec = None
        self.quality_maxgapsec = None

        self.gcmt_exclude = None
        self.gcmt_begin = None
        self.gcmt_end = None
        self.gcmt_minmag = None
        self.event_exclude_local_cat = None
        self.event_exclude_local_cat_begin = None
        self.event_exclude_local_cat_end = None
        self.event_exclude_local_cat_minmag = None
        self.event_exclude_local_cat_lat = None
        self.event_exclude_local_cat_lon = None
        self.event_exclude_local_cat_radius = None
        self.event_exclude = None
        self.event_exclude_winsec = None
        self.event_exclude_std = None
        self.event_exclude_n = None
        self.event_exclude_freqmin = None
        self.event_exclude_freqmax = None
        self.event_exclude_level = None
        
        self.interpolation_samples_gap = None

        self.wins = None
        self.wins_trim = None
        self.wins_detrend = None
        self.wins_demean = None
        self.wins_taper = None
        self.wins_taper_type = None
        self.wins_filter = None

        self.phaseshift = None
        self.Fs_old = None
        self.Fs_new = None
        self.Fs_antialias_factor = None

        self.instr_correction = None
        self.instr_correction_unit = None
        self.instr_correction_input = None
        self.instr_correction_prefilt = None
        self.instr_correction_waterlevel = None

        self.initialize()

    def initialize(self):
        """Populates the class from ./config.json.
        If ./config.json does not exist, writes a default file and exits.
        """
        CONFIG_Preprocess = self.CONFIG_Preprocess

        if not os.path.exists(CONFIG_Preprocess):
            with io.open(CONFIG_Preprocess, 'w') as fh:
                json.dump(DEFAULT_Preprocess, fh, sort_keys=True, indent=4, separators=(",", ": "))
            return()

        # Load all options.
        with io.open(CONFIG_Preprocess, 'r') as fh:
            data = json.load(fh)
            
        for key, value in data.items():
            setattr(self, key, value)
        
        # Make sure freqs. for downsampling are in descending order.
        self.Fs_new.sort() # Now in ascending order
        self.Fs_new=self.Fs_new[::-1] # Now in descending order


DEFAULT_Correlation = {
    "indirs": [],
    "time_begin": "2000-01-01T00:00:00.0000",
    "time_end": "2001-01-01T00:00:00.0000",
    "time_window_length":3600,
    "time_overlap":0.0,
    "time_min_window":3600,
    "corr_autocorr": False,
    "corr_only_autocorr": False,
    "corr_type": "ccc",
    "corr_maxlag": 0,
    "corr_normalize": True,
    "interm_stack": 0,
    "corr_tensorcomponents": ["ZZ"],
    "n_stationpairs": 1,
    "input_format": "MSEED",
    "format_output": "SAC",
    "locations_mix": False,
    "locations":[],
    "cap_glitch": False,
    "cap_thresh": 10.0,
    "bandpass": None,
    "whiten": False,
    "white_freqmin": 0.0,
    "white_freqmax": 0.0,
    "white_taper_samples":100,
    "onebit": False,
    "rotate": False,
    "ram_norm": False,
    "ram_window": 0.,
    "ram_prefilt": [],
    "update": False
}

#CONFIG_Correlation = os.path.join('input','config_correlation.json')

class ConfigCorrelation(object):
    """Contains basic parameters for the correlation job (paths, etc.)"""

    def __init__(self,CONFIG_Correlation):
        
        self.CONFIG_Correlation = CONFIG_Correlation
        self.project_path = None
        
        self.indirs = None
        self.bandpass = None
        self.cap_glitch = None
        self.cap_thresh = None
        self.time_begin = None
        self.time_end = None
        self.time_overlap = None
        self.time_window_length = None
        self.time_min_window = None
        self.corr_type = None
        self.corr_maxlag = None
        self.corr_tensorcomponents = None
        self.corr_autocorr = None
        self.corr_only_autocorr = None
        self.corr_normalize = None
        self.format_output = None
        self.input_format = None
        self.n_stationpairs = None
        self.interm_stack = None
        self.locations_mix = None
        self.locations = None
        self.update = None
        self.whiten = None
        self.white_freqmin = None
        self.white_freqmax = None
        self.white_taper_samples = None
        self.rotate = None
        self.ram_norm = None
        self.ram_window = None
        self.ram_prefilt = None
        self.onebit = None

        self.initialize()


    def initialize(self,check_params=True):

        """Populates the class from ./config.json.
        If ./config.json does not exist, writes a default file and exits."""

        CONFIG_Correlation = self.CONFIG_Correlation
        
        if not os.path.exists(CONFIG_Correlation):
            with io.open(CONFIG_Correlation, 'w') as fh:
                json.dump(DEFAULT_Correlation, fh, sort_keys=True, indent=4, separators=(",", ": "))
            return()

        # Load all options.
        with io.open(CONFIG_Correlation, 'r') as fh:
            data = json.load(fh)
            
        for key, value in data.items():
            setattr(self, key, value)
	
        if check_params:
            self.check_params()


    def check_params(self):
        chans = ["Z", "T", "R", "1", "2", "E", "N", "X", "Y"]
        components = []
        for c1 in chans:
            for c2 in chans:
                components.append(c1+c2)
        components = list(set(components))

        # lists

        if not isinstance(self.locations,list):
            msg = '\'locations\' in config_correlation.json must be list'
            raise TypeError(msg)

        if not isinstance(self.corr_tensorcomponents,list):
            msg = '\'corr_tensorcomponents\' in config_correlation.json must be list'
            raise TypeError(msg)

        if False in [(c in components) for c in self.corr_tensorcomponents]:
            msg = 'Tensor component not understood. Possible components are: {}'.format(components)
            raise ValueError(msg)


        if not isinstance(self.indirs,list):
            msg = '\'indirs\' in config_correlation.json must be list'
            raise TypeError(msg)

        if not True in [os.path.exists(d) for d in self.indirs]:
            msg = '\'indirs\' specified in config_correlation.json do not exist.'
            raise TypeError(msg)


        if self.bandpass is not None:
            if not isinstance(self.bandpass,list):
                
                msg = '\'bandpass\' in config_correlation.json must be list'
                raise TypeError(msg)
            else:
                #if self.bandpass is None: pass
                try:
                    bp = [float(n) for n in self.bandpass]
                except:
                    msg = '\'bandpass\' in config_correlation.json must be [freqmin, freqmax,order]'
                    raise ValueError(msg)


        if not isinstance(self.ram_prefilt,list):
            
            msg = '\'ram_prefilt\' in config_correlation.json must be list'
            raise TypeError(msg)
        else:
            
            try:
                bp = [float(n) for n in self.ram_prefilt]
            except:
                msg = '\'ram_prefilt\' in config_correlation.json must be [freqmin, freqmax,order]'
                raise ValueError(msg)




    # bools
    # bool

        if not isinstance(self.corr_autocorr,bool):
            msg = '\'corr_autocorr\' in config_correlation.json must be Boolean'
            raise TypeError(msg)

        if not isinstance(self.corr_normalize,bool):
            msg = '\'corr_normalize\' in config_correlation.json must be Boolean'
            raise TypeError(msg)

        if not isinstance(self.locations_mix,bool):
            msg = '\'locations_mix\' in config_correlation.json must be Boolean'
            raise TypeError(msg)

        if not isinstance(self.update,bool):
            msg = '\'update\' in config_correlation.json must be Boolean'
            raise TypeError(msg)

        if not isinstance(self.whiten,bool):
            msg = '\'whiten\' in config_correlation.json must be Boolean'
            raise TypeError(msg)

        if not isinstance(self.ram_norm,bool):
            msg = '\'ram_norm\' in config_correlation.json must be Boolean'
            raise TypeError(msg)

        if not isinstance(self.cap_glitch,bool):
            msg = '\'cap_glitch\' in config_correlation.json must be Boolean'
            raise TypeError(msg)

        if not isinstance(self.onebit,bool):
            msg = '\'onebit\' in config_correlation.json must be Boolean'
            raise TypeError(msg)

        
        try:
            t1 = UTCDateTime(self.time_begin)
        except:
            msg = 'Wrong format for \'time_begin\' in config_correlation.json.\nIt must be a string like 2000-01-01T00:00:00.0000.'
            raise ValueError(msg)

        try:
            t2 = UTCDateTime(self.time_end)
        except:
            msg = 'Wrong format for \'time_end\' in config_correlation.json.\nIt must be a string like 2000-01-01T00:00:00.0000.'
            raise ValueError(msg)

        if t2 < t1:
            msg = '\'time_end\' is before \'time_begin\'.'
            raise ValueError(msg)




    # Floats ======================================================================


        if not isinstance(self.cap_thresh,float):
            if not isinstance(self.cap_thresh,int):
                
                msg = '\'cap_thresh\' in config_correlation.json must be a positive number.'
                raise ValueError(msg)

        if not self.cap_thresh > 0:
            msg = '\'cap_thresh\' in config_correlation.json must be a positive number.'
            raise ValueError(msg)

        if not isinstance(self.time_overlap,float):
            if not isinstance(self.time_overlap,int):
                
                msg = '\'time_overlap\' in config_correlation.json must be a number.'
                raise ValueError(msg)
        
        if not isinstance(self.time_window_length,float):
            if not isinstance(self.time_window_length,int):
                
                msg = '\'time_window_length\' in config_correlation.json must be a number.'
                raise ValueError(msg)

        if not isinstance(self.time_min_window,float):
            if not isinstance(self.time_min_window,int):
                
                msg = '\'time_min_window\' in config_correlation.json must be a number.'
                raise ValueError(msg)


        if not isinstance(self.corr_maxlag,float):
            if not isinstance(self.corr_maxlag,int):
                
                msg = '\'corr_maxlag\' in config_correlation.json must be a number.'
                raise ValueError(msg)

        if not isinstance(self.white_freqmin,float):
            if not isinstance(self.white_freqmin,int):
                
                msg = '\'white_freqmin\' in config_correlation.json must be a number.'
                raise ValueError(msg)

        if not isinstance(self.white_freqmax,float):
            if not isinstance(self.white_freqmax,int):
                
                msg = '\'white_freqmax\' in config_correlation.json must be a number.'
                raise ValueError(msg)

        if not isinstance(self.ram_window,float):
            if not isinstance(self.ram_window,int):
                
                msg = '\'ram_window\' in config_correlation.json must be a number.'
                raise ValueError(msg)
            

            
        
            
            
          
            
            # string
            
        try:
            inp = str(self.input_format)
        except:
            
            msg = '\'input_format\' in config_correlation.json must be a string.'
            raise ValueError(msg)

            # integer
        if not isinstance(self.n_stationpairs,int):
            msg = '\'n_stationpairs\' in config_correlation.json must be integer.'
            raise ValueError(msg)

        if not isinstance(self.interm_stack,int):
            msg = '\'interm_stack\' in config_correlation.json must be integer.'
            raise ValueError(msg)

        if not isinstance(self.white_taper_samples,int):
            msg = '\'white_taper_samples\' in config_correlation.json must be integer.'
            raise ValueError(msg)


