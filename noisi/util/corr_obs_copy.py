import numpy as np
import os
from pandas import read_csv
from shutil import copy2
import obspy
import csv
import os
import io
import yaml
from noisi.util.windows import get_window
from noisi.util.source_grid_svp import spherical_distance_degrees

import functools
print = functools.partial(print, flush=True)

def copy_corr(args,comm,size,rank):
    """
    This function is used to copy observed cross-correlations based on a signal-to-noise ratio.
    
    args: project_path, observed_corr, source_model, args.snr_thresh
    """
    
    stationlist_path = os.path.join(args.project_path,"stationlist.csv")
    observed_correlations = args.observed_corr
    
    # measr_config for window signal to noise ratio
    measr_configfile = os.path.join(args.source_model,
                                    'measr_config.yml')
    measr_config = yaml.safe_load(open(measr_configfile))
    
    # parameters
    if measr_config['bandpass'] is not None:
        bandpass = measr_config['bandpass'][0]
    
    params = {'g_speed': measr_config["g_speed"],
              'hw': measr_config["window_params_hw"],
              'hw_variable': measr_config["window_params_hw_variable"],
              'sep_noise': measr_config["window_params_sep_noise"],
              'wtype': 'boxcar',
              'win_overlap': measr_config["window_params_win_overlap"]
             }
    
    
    
    # output folder
    corr_obs_path = os.path.join(args.source_model,'observed_correlations')
    corr_obs_path_slt = os.path.join(args.source_model,'observed_correlations_slt')
    
    # input folder
    obs_files = [os.path.join(observed_correlations,i) for i in os.listdir(observed_correlations) if i.endswith('.SAC') or i.endswith('.sac')]
    
    if rank == 0:
        print(f'There are {np.size(obs_files)} observed correlations.')

    # split up indices for mpi
    nr_corr = np.size(obs_files)
    nr_pr = int(np.ceil(nr_corr/size))
    corr_ind = np.arange(rank*nr_pr,(rank+1)*nr_pr)

    
    # make set with station pairs
    if args.opt_statpair is not None:
        if rank == 0:
            print("Using optimal station pair file..")
        
        opt_statpair_set = set(read_csv(args.opt_statpair,keep_default_na=False)['sta_pair'])
        
        for k in range(nr_pr):
            # avoid index error
            if corr_ind[k] >= nr_corr:
                break

            j = os.path.basename(obs_files[corr_ind[k]])
        
            statpair_name = j[:-4]
            
            if statpair_name in opt_statpair_set:
                src = os.path.join(observed_correlations,j)
                dst = os.path.join(corr_obs_path,j)
                copy2(src,dst)
                                        
                if corr_ind[k]%args.frac_corr_slt == 0:

                    src = os.path.join(observed_correlations,j)
                    dst = os.path.join(corr_obs_path_slt,j)
                    copy2(src,dst)
                
            
    else:
        # first make dictionary with station distances
        stationlist = read_csv(stationlist_path,keep_default_na=False)
        station_dict_dist = dict()

        for i in range(0,np.size(stationlist,0)):
            for j in range(0,np.size(stationlist,0)):

                net_1 = stationlist.at[i,'net']
                sta_1 = stationlist.at[i,'sta']
                net_2 = stationlist.at[j,'net']
                sta_2 = stationlist.at[j,'sta']

                lat_1 = stationlist.at[i,'lat']
                lon_1 = stationlist.at[i,'lon']
                lat_2 = stationlist.at[j,'lat']
                lon_2 = stationlist.at[j,'lon']

                stat_pair_name = str(net_1) + '.' + str(sta_1) + '--' + str(net_2) + '.' + str(sta_2)

                dist = spherical_distance_degrees(lat_1,lon_1,lat_2,lon_2)
                station_dict_dist.update({stat_pair_name:dist})

        counter = 0
        count = 0
        count_tot = 0

        for k in range(nr_pr):
            # avoid index error
            if corr_ind[k] >= nr_corr:
                break

            j = obs_files[corr_ind[k]]

            #if k%1000 == 0:
            #    print(f'Rank {rank}: {k} of {nr_pr} correlations.')

            s_var = obspy.read(j)
            
            if measr_config['bandpass'] is not None:
                tr = s_var[0].filter('bandpass',freqmin=bandpass[0],freqmax=bandpass[1],corners=bandpass[2],zerophase=True)
            else:
                tr = s_var[0]
                
            t_min = int(np.floor((tr.stats.npts*tr.stats.delta)/2))
            data_var = tr.data
            data_t = np.linspace(-t_min,t_min,np.size(data_var))

            net_1 = os.path.basename(j).split('.')[0]
            sta_1 = os.path.basename(j).split('.')[1]

            net_2 = os.path.basename(j).split('.')[3].split('--')[1]
            sta_2 = os.path.basename(j).split('.')[4]

            stat_pair_name = net_1 + '.' + sta_1 + '--' + net_2 + '.' + sta_2

            if stat_pair_name in station_dict_dist.keys():
                #print('Only copying correlations from stationlist')
                # compute expected arrival time
                # should probably be read from source_config or measr_config
                
                if args.snr_thresh == 0 or args.snr_thresh == None:

                    if args.corr_max_dist is not None:
                        
                        if station_dict_dist[stat_pair_name] < args.corr_max_dist: 
                        
                            i = os.path.basename(j)
                            src = os.path.join(observed_correlations,i)
                            dst = os.path.join(corr_obs_path,i)
                            copy2(src,dst)
                            count += 1

                            if corr_ind[k]%args.frac_corr_slt == 0:
                                src = os.path.join(observed_correlations,i)
                                dst = os.path.join(corr_obs_path_slt,i)
                                copy2(src,dst)
                        
                        else:
                            continue
                
                    else:
                        i = os.path.basename(j)
                        src = os.path.join(observed_correlations,i)
                        dst = os.path.join(corr_obs_path,i)
                        copy2(src,dst)
                        count += 1

                        if corr_ind[k]%args.frac_corr_slt == 0:
                            src = os.path.join(observed_correlations,i)
                            dst = os.path.join(corr_obs_path_slt,i)
                            copy2(src,dst)
                else:
                    
                    # get windows
                    win_signal, win_noise, scs = get_window(tr.stats,g_speed=params['g_speed'],params=params)
                    
                    # only use if expected arrival time window is included
                    if scs:
                        
                        corr_data = data_var
                        corr_data_norm = corr_data/np.max(np.abs(corr_data))
                        
                        # make sure it's boxcar window
                        win_signal[win_signal>0] = 1
                        
                        # compute signal to noise ratio
                        corr_data_win = data_var*win_signal + data_var*np.flip(win_signal)
                        corr_data_win_norm = corr_data_win/np.max(np.abs(corr_data))

                        corr_std = np.std(corr_data_norm)
                        
                        # get maximum only in windows
                        corr_max = np.max(np.abs(corr_data_win_norm))
                        std_max_var = corr_max/corr_std
                        
                        
                        if args.corr_max_dist is not None:
                            if station_dict_dist[stat_pair_name] < args.corr_max_dist and std_max_var > args.snr_thresh:  

                                i = os.path.basename(j)
                                src = os.path.join(observed_correlations,i)
                                dst = os.path.join(corr_obs_path,i)
                                copy2(src,dst)
                                count += 1

                                if corr_ind[k]%args.frac_corr_slt == 0:

                                    src = os.path.join(observed_correlations,i)
                                    dst = os.path.join(corr_obs_path_slt,i)
                                    copy2(src,dst)

                            else:
                                continue

                        elif std_max_var > args.snr_thresh:

                            i = os.path.basename(j)
                            src = os.path.join(observed_correlations,i)
                            dst = os.path.join(corr_obs_path,i)
                            copy2(src,dst)
                            count += 1

                            if corr_ind[k]%args.frac_corr_slt == 0:

                                src = os.path.join(observed_correlations,i)
                                dst = os.path.join(corr_obs_path_slt,i)
                                copy2(src,dst)

                        else:
                            continue
                            
                    else:
                        continue

                counter += 1
                count_tot += 1
            else:
                count_tot += 1
                pass

    comm.barrier()

    corr_obs_files =   [os.path.join(corr_obs_path,i) for i in os.listdir(corr_obs_path) if i.endswith('.SAC') or i.endswith('.sac')]

    if rank == 0:
        print(f'Copied {np.size(os.listdir(corr_obs_path))} files to {corr_obs_path}.')
        print(f'Copied {np.size(os.listdir(corr_obs_path_slt))} files to {corr_obs_path_slt}.')

    
    return np.size(os.listdir(corr_obs_path))