"""
!!!!! NOT NEEDED FOR NOISI !!!!!

Create synthetic data for stationary phase analysis

:copyright:
    noisi development team
:license:
    GNU Lesser General Public License, Version 3 and later
    (https://www.gnu.org/copyleft/lesser.html)
"""


from obspy.geodetics import gps2dist_azimuth
from obspy.geodetics import kilometers2degrees
from obspy.taup import TauPyModel
import obspy
import numpy as np
from pandas import read_csv
import os

def create_synth(args,comm,size,rank):
    """
    Function to create synthetic correlations for stationary phases analysis
    """
    
    # set taup model
    model = TauPyModel(model=args.taup_model)
    
    
    # first get all possible station pairs and their distances
    stationlist = read_csv(args.stationlist_path,keep_default_na = False)
    stat_lat = stationlist['lat']
    stat_lon = stationlist['lon']

    stat_dict = dict()
    stat_pair = dict()

    for i in stationlist.iterrows():

        net_1,sta_1,lat_1,lon_1 = i[1]['net'],i[1]['sta'],i[1]['lat'],i[1]['lon']
        stat_dict.update({f"{net_1}.{sta_1}":[lat_1,lon_1]})

        for j in stationlist.iterrows():
    
            net_2,sta_2,lat_2,lon_2 = j[1]['net'],j[1]['sta'],j[1]['lat'],j[1]['lon']
            
            if args.correlation_path is None:
                # ignore autocorr
                if net_1 == net_2 and sta_1 == sta_2:
                    continue
                
                # don't do it in both directions
                if f"{net_1}.{sta_1}--{net_2}.{sta_2}" in list(stat_pair.keys()) or f"{net_2}.{sta_2}--{net_1}.{sta_1}" in list(stat_pair.keys()):
                    continue

            dist_var,az_var, baz_var = gps2dist_azimuth(lat_1,lon_1,lat_2,lon_2)
            #dist_deg_var = kilometers2degrees(dist_var)
            
            
            stat_pair.update({f"{net_1}.{sta_1}--{net_2}.{sta_2}":[dist_var,az_var,baz_var]})
    
    
    # if correlation path is given, only use those
    if args.correlation_path is not None:
        args.correlation_path = os.path.abspath(args.correlation_path)    
    
        corr_files = [file for file in os.listdir(args.correlation_path) if file.split(".")[-1] in args.corr_format]
        
        stat_pair_list = []
        
        for corr in corr_files:
            net_1 = corr.split('--')[0].split('.')[0]
            sta_1 = corr.split('--')[0].split('.')[1]
            net_2 = corr.split('--')[1].split('.')[0]
            sta_2 = corr.split('--')[1].split('.')[1]
            
            stat_pair_list.append(f"{net_1}.{sta_1}--{net_2}.{sta_2}")
    
    else:
        stat_pair_list = list(stat_pair.keys())
    
    # split list of stat_pairs up for different ranks
    if rank == 0:
        stat_pair_split = np.array_split(stat_pair_list,size)
        stat_pair_split = [k.tolist() for k in stat_pair_split]
    else:
        stat_pair_split = None
    
    stat_pair_split = comm.scatter(stat_pair_split,root=0) 
            
    
    # iterate over the split up list and create synthetics
    for k,i in enumerate(stat_pair_split):
        
        if k%100 == 0 and rank == 0:
            print(f"At {k} of {np.size(stat_pair_split)} station pairs on rank {rank}")
        
        stat_1 = i.split('--')[0]
        stat_2 = i.split('--')[1]

        # create a trace
        tr_var = obspy.Trace()
        
        tr_var.data = np.zeros(args.stat_phase_npts)
        tr_var.stats.sampling_rate = 1/args.stat_phase_dt
        
        # add metadata
        tr_var.stats.network = stat_1.split('.')[0]
        tr_var.stats.station = stat_1.split('.')[1]
        tr_var.stats.location = ''
        tr_var.stats.channel = 'MXZ'
        
        tr_var.stats.sac = {}
        tr_var.stats.sac.stla = stat_dict[stat_1][0]
        tr_var.stats.sac.stlo = stat_dict[stat_1][1]
        tr_var.stats.sac.evla = stat_dict[stat_2][0]
        tr_var.stats.sac.evlo = stat_dict[stat_2][1]
        tr_var.stats.sac.dist = stat_pair[i][0]
        tr_var.stats.sac.az = stat_pair[i][1]
        tr_var.stats.sac.baz = stat_pair[i][2]
        tr_var.stats.sac.npts = args.stat_phase_npts
        tr_var.stats.sac.kstnm = stat_1.split('.')[1]
        tr_var.stats.sac.kevnm = stat_2.split('.')[1]
        tr_var.stats.sac.kuser0 = stat_2.split('.')[0]
        tr_var.stats.sac.kuser1 = ''
        tr_var.stats.sac.kuser2 = 'MXZ'
        tr_var.stats.sac.kcmpnm = 'MXZ'
        tr_var.stats.sac.knetwk = stat_2.split('.')[0]
        
        
        # Add Gauss or Ricker wavelet at certain distance
        # iterate over different main phases
        
        for phase in args.main_phases:
            
            # get arrival time by taking one station as source
            dist_deg= kilometers2degrees(stat_pair[i][0]/1000)
            
            arrivals = model.get_travel_times(source_depth_in_km=0,distance_in_degree=dist_deg,phase_list=[phase])
            
            # Use the first arrival
            arr_1 = arrivals[0].time
            #print(arr_1)
        
            if args.stat_phase_input.lower() in ['gauss','gaussian']:
                
                # get the timeshift
                time_shift = (np.size(tr_var.data))/2
                time = np.linspace(-time_shift,time_shift,np.size(tr_var.data))

                # Add it to other causal, acausal, or both
                if args.stat_phase_caus.lower() in ['caus','causal']:
                    arr_time = [arr_1]
                    
                elif args.stat_phase_caus.lower() in ['acaus','acausal']:
                    arr_time = [-arr_1]
                    
                elif args.stat_phase_caus.lower() == 'both':
                    arr_time = [arr_1,-arr_1]

                for arr in arr_time:
                    spec = np.exp(-((time - arr) ** 2) /  (2 * args.stat_phase_sigma ** 2))
                    tr_var.data += spec
                    
                    
            if args.stat_phase_input.lower() in ['ricker','rick']:
                from scipy.signal import ricker
                
                # get the timeshift
                time_shift = (np.size(tr_var.data))/2
                time = np.linspace(-time_shift,time_shift,np.size(tr_var.data))

                # Add it to other causal, acausal, or both
                if args.stat_phase_caus.lower() in ['caus','causal']:
                    arr_time = [arr_1]
                    
                elif args.stat_phase_caus.lower() in ['acaus','acausal']:
                    arr_time = [-arr_1]
                    
                elif args.stat_phase_caus.lower() == 'both':
                    arr_time = [arr_1,-arr_1]

                for arr in arr_time:
                    # create ricker and roll to correct lag
                    spec = np.roll(ricker(np.size(tr_var.data),args.stat_phase_sigma),int(arr*args.stat_phase_dt))
                    tr_var.data += spec
                
                
                    
            # write the correlation                  
            tr_var_name = f'{tr_var.stats.network}.{tr_var.stats.station}..{tr_var.stats.channel}--{tr_var.stats.sac.knetwk}.{tr_var.stats.sac.kevnm}..{tr_var.stats.sac.kcmpnm}'
            
            tr_var_path = os.path.join(args.project_path,f'corr_stat_phase_{phase}_{args.stat_phase_input}')
            
            tr_var.write(os.path.join(tr_var_path,f'{tr_var_name}.sac'))
        
    return
