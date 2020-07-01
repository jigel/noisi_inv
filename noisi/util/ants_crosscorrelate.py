"""
This script uses functions from the ants module: https://github.com/lermert/ants_2.
Previously downloaded data is processed and cross-correlated.
"""

import numpy as np
import os
import obspy
from pandas import read_csv
from glob import glob
from noisi.ants.scripts.ant_preprocess_noisi import preprocess
from noisi.ants.scripts.ant_correlation_noisi import correlate
from noisi.ants.config_noisi import ConfigPreprocess
from noisi.ants.config_noisi import ConfigCorrelation
import json

import functools
print = functools.partial(print, flush=True)

def ants_preprocess(args,comm,size,rank):
    """
    Config file is created in here.
    To change parameters for preprocessing, this script has to be modified
    """
   
    if rank == 0:
        print("Pre-processing data...")   


    data_raw_path = os.path.join(args.project_path,'data','raw')

    # config file created here. Change parameters
    config_preprocess = {
        "project_path": args.project_path,

        "Fs_antialias_factor": 0.4,
        "Fs_new": [
            20,
            10,
            1
        ],
        "Fs_old": [
            200,
            120,
            100,
            50,
            40,
            30,
            20,
            10
        ],
        "event_exclude": False,
        "event_exclude_freqmax": 1.0,
        "event_exclude_freqmin": 0.01,
        "event_exclude_level": 2.0,
        "event_exclude_local_cat": False,
        "event_exclude_local_cat_begin": str(args.t_start).split('-')[0]+','+str(args.t_start).split('-')[1]+','+str(args.t_start).split('-')[2].split('T')[0],
        "event_exclude_local_cat_end": str(args.t_end).split('-')[0]+','+str(args.t_end).split('-')[1]+','+str(args.t_end).split('-')[2].split('T')[0],
        "event_exclude_local_cat_lat": 0.0,
        "event_exclude_local_cat_lon": 0.0,
        "event_exclude_local_cat_minmag": 2.0,
        "event_exclude_local_cat_radius": 10.0,
        "event_exclude_n": 4,
        "event_exclude_std": 2.0,
        "event_exclude_winsec": [],
        "gcmt_begin": str(args.t_start).split('-')[0]+','+str(args.t_start).split('-')[1]+','+str(args.t_start).split('-')[2].split('T')[0],
        "gcmt_end": str(args.t_end).split('-')[0]+','+str(args.t_end).split('-')[1]+','+str(args.t_end).split('-')[2].split('T')[0],
        "gcmt_exclude": False,
        "input_dirs": [
            f"{data_raw_path}"
        ],
        "input_format": "MSEED",
        "instr_correction": True,
        "instr_correction_input": "staxml",
        "instr_correction_prefilt": [
            0.01,
            0.02,
            5,
            5.2
        ],
        "instr_correction_unit": args.synt_data,
        "instr_correction_waterlevel": 0.0,
        "locations": [
            "",
            "00",
            "01",
            "10",
            "11",
        ],
        "quality_maxgapsec": 120.0,
        "quality_minlengthsec": 0.0,
        "testrun": False,
        "verbose": True,
        "wins": True,
        "wins_demean": True,
        "wins_detrend": True,
        "wins_len_sec": 3600,
        "wins_taper": 0.01,
        "wins_taper_type": "cosine",
        "wins_trim": True
    }

    # save to config file
    config_preprocess_file = os.path.join(args.project_path,'config_preprocess.json')
    
    if rank == 0:
        with open(config_preprocess_file, 'w') as outfile:
            json.dump(config_preprocess, outfile,indent=2)

    comm.barrier()

    # preprocess using ants
    cfg_preprocess = ConfigPreprocess(config_preprocess_file)

    preprocess(cfg_preprocess,comm,size,rank)
    
    comm.barrier()
    
    if rank == 0:
        print("Pre-processing done.")
    
    return


def ants_crosscorrelate(args,comm,size,rank):
    """
    Config file is created in here.
    To change parameters for cross-correlating, this script has to be modified
    """
    
    if rank == 0:
        print("Cross-correlating data...")
    
    data_processed_path = os.path.join(args.project_path,'data','processed')
    
    # cross-correlation channels
    if args.wavefield_channel == 'all':
        corr_comp = ['ZZ', 'RR', 'TT', 'ZR', 'RZ', 'ZT', 'TZ', 'RT', 'TR']
    elif args.wavefield_channel == 'Z':
        corr_comp = 'ZZ'
    elif args.wavefield_channel == 'E':
        corr_comp = 'RR'
    elif args.wavefield_channel == 'N':
        corr_comp = 'TT'
    
    # cross-correlate
    config_correlation = {
        "project_path": args.project_path,

        "bandpass": None,
        "cap_glitch": False,
        "cap_thresh": 10.0,
        "corr_autocorr": False,
        "corr_maxlag": args.max_lag,
        "corr_normalize": False,
        "corr_tensorcomponents": corr_comp,
        "corr_type": "ccc",
        "format_output": "SAC",
        "indirs": [
            data_processed_path
        ],
        "input_format": "MSEED",
        "interm_stack": 0,
        "locations": [
            "",
            "00",
            "01",
            "10",
            "11",
        ],
        "locations_mix": True,
        "n_stationpairs": 1,
        "onebit": False,
        "ram_norm": False,
        "ram_prefilt": [],
        "ram_window": 0.0,
        "time_begin": str(args.t_start),
        "time_end": str(args.t_end),
        "time_min_window": 3600,
        "time_overlap": 0.0,
        "time_window_length": 3600,
        "update": False,
        "white_freqmax": 0.0,
        "white_freqmin": 0.0,
        "white_taper_samples": 100,
        "whiten": False
    }

    # save to config file
    config_correlation_file = os.path.join(args.project_path,'config_correlation.json')

    if rank == 0:
        with open(config_correlation_file, 'w') as outfile:
            json.dump(config_correlation, outfile,indent=2)
        
    comm.barrier()

    cfg_corr = ConfigCorrelation(config_correlation_file)

    correlate(cfg_corr,comm,size,rank)

    comm.barrier()
    
    if rank == 0:
        print("Cross-correlating done.")
        
    return 