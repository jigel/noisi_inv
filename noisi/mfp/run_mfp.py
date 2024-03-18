"""
Matched-Field Processing code to compute an initial model

:copyright:
    noisi development team
:license:
    GNU Lesser General Public License, Version 3 and later
    (https://www.gnu.org/copyleft/lesser.html)
"""


import numpy as np
import yaml 
import os
import sys
import shutil
from glob import glob
from pandas import read_csv
import matplotlib
matplotlib.use('Agg')
import time
import psutil
process = psutil.Process(os.getpid())
import functools
print = functools.partial(print, flush=True)

# mfp codes
from noisi.mfp.mfp_code.scripts.create_sourcegrid import create_sourcegrid
from noisi.mfp.mfp_code.util.plot import plot_grid
from noisi.mfp.mfp_code.scripts.create_stat_phase_synthetics import create_synth
from noisi.mfp.mfp_code.scripts.mfp_main import mfp_main

# plotting
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.tri as tri
from matplotlib import colors
import cmasher as cmr
cmap = cmr.cosmic    # CMasher
cmap = plt.get_cmap('cmr.cosmic')   # MPL



#from mpi4py import MPI
# simple embarrassingly parallel run:
#comm = MPI.COMM_WORLD
#size = comm.Get_size()
#rank = comm.Get_rank()

t_0 = time.time()



# use a class args to give values to different codes
class mfp_args_1(object):
    
    def __init__(self):
        
        self.project_name = None
        self.output_path = None
        self.project_path = None
        self.correlation_path = None
        self.corr_format = None
        self.stationlist_path = None
        self.station_distance_min = None
        self.station_distance_max = None
        self.sourcegrid_path = None
        self.svp_grid_config = None
        self.method = None
        self.stationary_phases = None
        self.stat_phase_input = None 
        self.bandpass_filter = [0.1,0.2,5]
        self.taup_model = 'iasp91'
        self.phases = ['3kmps','P']
        self.plot = True
        
        
def run_noisi_mfp(args,comm,size,rank):
    # Read in the mfp_config.yml file which should be given as input after mfp code
    mfp_args = mfp_args_1()
    
    mfp_config_path = os.path.join(args.project_path,'mfp_config.yml')

    
    with open(mfp_config_path) as f:
        mfp_config = yaml.safe_load(f)

    for attr in mfp_config:
        setattr(mfp_args,attr,mfp_config[attr])


    # perform some checks before starting
    if not mfp_args.stationary_phases and mfp_args.correlation_path is None:
        raise Exception('Need to set correlation path if stationary_phases is False.')

    if mfp_args.stationary_phases and not isinstance(mfp_args.phase_pairs,list) and len(mfp_args.phases) == 1 and not mfp_args.phase_pairs_auto and mfp_args.phase_pairs != 'same':
        raise Exception('For stationary phases set either multiple phases or give more than one phase in phase_pairs')

    mfp_args.output_path = os.path.abspath(mfp_args.output_path)
    mfp_args.stationlist_path = os.path.abspath(mfp_args.stationlist_path)


    # Make project folder and copy files there
    if rank == 0:
        print("===="*20)
        print(f"Running MFP to use as starting model")
        print("===="*20)


    # boolean used to check grid validity for all ranks
    grid_val = True


    # make project folder
    if rank == 0:
        if not os.path.isdir(os.path.abspath(os.path.join(mfp_args.output_path,mfp_args.project_name))):
            os.makedirs(os.path.abspath(os.path.join(mfp_args.output_path,mfp_args.project_name)))

    comm.Barrier()

    # make sure it's set on all ranks 
    mfp_args.project_path = os.path.abspath(os.path.join(mfp_args.output_path,mfp_args.project_name))

    if rank == 0:
        print("Project path: ", mfp_args.project_path)

        # copy sourcegrid and stationlist
        shutil.copy(mfp_args.stationlist_path,os.path.join(mfp_args.project_path,'stationlist.csv'))
        mfp_args.stationlist_path = os.path.join(mfp_args.project_path,'stationlist.csv')

        # copy mfp_config
        shutil.copy(mfp_config_path,os.path.join(mfp_args.project_path,'mfp_config.yml'))

        # copy sourcegrid file or create new grid
        if mfp_args.sourcegrid_path.endswith('.npy'):

            shutil.copy(mfp_args.sourcegrid_path,os.path.join(mfp_args.project_path,'sourcegrid.npy'))
            mfp_args.sourcegrid_path = os.path.join(mfp_args.project_path,'sourcegrid.npy')

            sourcegrid = np.load(mfp_args.sourcegrid_path)

            plot_grid(mfp_args,grid=[sourcegrid[1],sourcegrid[0]],
                      output_file=os.path.join(mfp_args.project_path,'sourcegrid.png'),
                      only_ocean=False,
                      title=f'Sourcegrid with {np.size(sourcegrid[0])} gridpoints',
                      stationlist_path=mfp_args.stationlist_path)

        elif mfp_args.sourcegrid_path.lower() in ['svp_grid','svp','grid','svp_grids']:
            print("Creating new sourcegrid using the given parameters.")

            # create new svp grid
            sourcegrid = create_sourcegrid(mfp_args.svp_grid_config)
            # number of gridpoints
            print(f"Number of gridpoints: {np.size(sourcegrid[0])}")

            # save grid as npy file
            np.save(os.path.join(mfp_args.project_path,'sourcegrid.npy'),sourcegrid)

            mfp_args.sourcegrid_path = os.path.join(mfp_args.project_path,'sourcegrid.npy')

            # plot sourcegrid
            plot_grid(mfp_args,grid=[sourcegrid[1],sourcegrid[0]],
                      data=None,
                      output_file=os.path.join(mfp_args.project_path,'sourcegrid.png'),
                      only_ocean=False,title=f'Sourcegrid with {np.size(sourcegrid[0])} gridpoints',
                      stationlist_path=mfp_args.stationlist_path)


        else: 
            grid_val = False


    comm.Barrier()

    # set sourcegrid path on all rank
    mfp_args.sourcegrid_path = os.path.join(mfp_args.project_path,'sourcegrid.npy')


    # exit all ranks if grid not valid
    grid_val = comm.bcast(grid_val,root=0)
    if not grid_val:
        raise Exception("No valid sourcegrid input given. Has to be either .npy file or svp_grid.")


    if rank == 0:
        run_time = open(os.path.join(mfp_args.project_path,'runtime.txt'),'w+')
        run_time.write(f"Number of cores: {size} \n")
        run_time.write(f"Number of grid points: {np.size(sourcegrid[0])}\n")
        t_1 = time.time()
        run_time.write(f"Project setup: {np.around((t_1-t_0)/60,4)} min \n")

        #### Check memory usage
        #print("Memory usage in Gb at start ", process.memory_info().rss / 1.e9)
        run_time.write(f"Memory usage in Gb at start {process.memory_info().rss / 1.e9} \n")


    comm.Barrier()

    # If stationary phases are to be calculated, create a synthetic correlation data with Ricker/Gauss wavelet at arrival time
    if mfp_args.stationary_phases:

        # check which phases are wanted
        phase_pair_list = []

        if mfp_args.phase_pairs == 'all':
            if mfp_args.phase_pairs_auto:
                phase_pair_list = [[i,j] for i in mfp_args.phases for j in mfp_args.phases]
            else:
                phase_pair_list = [[i,j] for i in mfp_args.phases for j in mfp_args.phases if i != j]

        elif mfp_args.phase_pairs == 'same':
            phase_pair_list = [[i,j] for i in mfp_args.phases for j in mfp_args.phases if i == j]

        elif isinstance(mfp_args.phase_pairs,list):
            phase_pair_list = []

            for pair in mfp_args.phase_pairs:
                list_1 = [i for i in pair.split('-')] # one way
                list_2 = [i for i in pair.split('-')[::-1]] # and the other way too

                if list_1 in phase_pair_list:
                    continue
                else:
                    phase_pair_list.append(list_1)

                # ignore when it's the same, e.g. P-P or PKP-PKP
                if list_1 == list_2:
                    continue

                if list_2 in phase_pair_list:
                    continue
                else:
                    phase_pair_list.append(list_2)


        mfp_args.phase_list = phase_pair_list
        # get the main phases to create synthetic data, i.e. the ones with only one letter
        mfp_args.main_phases = mfp_args.stat_phase_main

        if rank == 0:
            print(f"Creating synthetic correlations for stationary phase analysis: {mfp_args.main_phases}")

            # create folder for these correlations
            for phase in mfp_args.main_phases:
                mfp_args.corr_stat_phase_path = os.path.join(mfp_args.project_path,f'corr_stat_phase_{phase}_{mfp_args.stat_phase_input.lower()}')

                if not os.path.isdir(mfp_args.corr_stat_phase_path):
                    os.makedirs(mfp_args.corr_stat_phase_path)

        comm.Barrier()

        create_synth(mfp_args,comm,size,rank)

        if rank == 0:
            print("All synthetic correlations created.")


        comm.Barrier()

    # if it's not a stationary phase run, make phase list from normal phases
    else:
        mfp_args.correlation_path = os.path.abspath(mfp_args.correlation_path)

        # check which phases are wanted
        phase_pair_list = []

        if mfp_args.phase_pairs == 'all':
            phase_pair_list = [[i,j] for i in mfp_args.phases for j in mfp_args.phases]

        elif mfp_args.phase_pairs == 'same':
            phase_pair_list = [[i,j] for i in mfp_args.phases for j in mfp_args.phases if i == j]

        elif isinstance(mfp_args.phase_pairs,list):
            phase_pair_list = []

            for pair in mfp_args.phase_pairs:
                list_1 = [i for i in pair.split('-')] # one way
                list_2 = [i for i in pair.split('-')[::-1]] # and the other way too

                if list_1 in phase_pair_list:
                    continue
                else:
                    phase_pair_list.append(list_1)

                if list_2 in phase_pair_list:
                    continue
                else:
                    phase_pair_list.append(list_2)

        # To make sure it doesn't loop 
        mfp_args.main_phases = ['corr']
        mfp_args.phase_list = phase_pair_list



    if rank == 0:
        print(f"Phase pairs: {phase_pair_list}")

        t_2 = time.time()
        run_time.write(f"Before MFP: {np.around((t_2-t_1)/60,4)} min \n")

        #### Check memory usage
        #print("Memory usage in Gb before MFP ", process.memory_info().rss / 1.e9)
        run_time.write(f"Memory usage in Gb before MFP {process.memory_info().rss / 1.e9} \n")


    ### DO MFP
    ## Instead of iterating over the grid, iterate over the correlations
    ## Should create a map for each phase pair
    # iterate over the main phases
    

    # for stationary phases loop over the main_phases
    for phase in mfp_args.main_phases:

        if rank == 0 and mfp_args.stationary_phases:
            print(f"Working on phase {phase} arrivals..")

        if mfp_args.stationary_phases:
            mfp_args.correlation_path = os.path.join(mfp_args.project_path,f'corr_stat_phase_{phase}_{mfp_args.stat_phase_input.lower()}')

        # run matched field processing
        mfp = mfp_main(mfp_args,comm,size,rank)


        # save the different mfp maps and plot
        if rank == 0:
            print("Saving and plotting output..")

            if not os.path.isdir(os.path.join(mfp_args.project_path,"mfp_results")):
                os.makedirs(os.path.join(mfp_args.project_path,"mfp_results"))

            mfp_result_path = os.path.join(mfp_args.project_path,"mfp_results")

            if not os.path.isdir(os.path.join(mfp_args.project_path,"mfp_plots")):
                os.makedirs(os.path.join(mfp_args.project_path,"mfp_plots"))

            mfp_plot_path = os.path.join(mfp_args.project_path,"mfp_plots")


            # sum them up if that's wanted
            if mfp_args.phase_pairs_sum:
                mfp_sum = dict()

                
            for i,phases in enumerate(mfp):

                for m_idx,method in enumerate(mfp_args.method):
                    meth_idx = m_idx+2            

                    # save 
                    if not os.path.isfile(os.path.join(mfp_result_path,f'MFP_{phase}_{phases}_{method}.npy')):
                        np.save(os.path.join(mfp_result_path,f'MFP_{phase}_{phases}_{method}.npy'),mfp[phases][meth_idx])
                    else:
                        np.save(os.path.join(mfp_result_path,f'MFP_{phase}_{phases}_{method}_{i}.npy'),mfp[phases][meth_idx])


                    if mfp_args.phase_pairs_sum and i == 0:
                        mfp_sum['grid'] = [mfp[phases][0],mfp[phases][1]]
                        mfp_sum[method] = mfp[phases][meth_idx]
                    elif mfp_args.phase_pairs_sum:
                        mfp_sum[method] += mfp[phases][meth_idx]



                    if mfp_args.plot:

                        if not os.path.isfile(os.path.join(mfp_plot_path,f'MFP_{phase}_{phases}_{method}.png')):
                            output_file = os.path.join(mfp_plot_path,f'MFP_{phase}_{phases}_{method}.png')
                        else:
                            output_file = os.path.join(mfp_plot_path,f'MFP_{phase}_{phases}_{method}_{i}.png')


                        plot_grid(mfp_args,grid=[mfp[phases][1],mfp[phases][0]],
                                  data=mfp[phases][meth_idx],
                                  output_file=output_file,
                                  triangulate=True,
                                  cbar=True,
                                  only_ocean=False,
                                  title=f'MFP for phases {phases} using {phase} arrival. Method: {method}.',
                                  stationlist_path=mfp_args.stationlist_path)



            if mfp_args.phase_pairs_sum:

                for method in mfp_args.method:
                    np.save(os.path.join(mfp_result_path,f'MFP_sum_{phase}_{method}.npy'),mfp_sum[method])

                    plot_grid(mfp_args,grid=[mfp_sum['grid'][1],mfp_sum['grid'][0]],
                              data=mfp_sum[method],
                              output_file=os.path.join(mfp_plot_path,f'MFP_sum_{phase}_{method}.png'),
                              triangulate=True,
                              cbar=True,
                              only_ocean=False,
                              title=f'MFP for all phases using {phase} arrival. Method: {method}.',
                              stationlist_path=mfp_args.stationlist_path)


    comm.Barrier()


    if rank == 0:
        t_3 = time.time()
        run_time.write(f"After MFP: {np.around((t_3-t_2)/60,4)} min \n")
        run_time.write(f"Total runtime: {np.around((t_3-t_0)/60,4)} min \n")

        #### Check memory usage
        #print("Memory usage in Gb after MFP ", process.memory_info().rss / 1.e9)
        run_time.write(f"Memory usage in Gb after MFP {process.memory_info().rss / 1.e9} \n")
        run_time.close()

        print("===="*20)
        print(f"MFP done.")
        #print(f"Results in {mfp_result_path}")
        #print(f"Plots in {mfp_plot_path}")
        print("===="*20)

    comm.Barrier()
    
    
    for i,phases in enumerate(mfp):
        for m_idx,method in enumerate(mfp_args.method):
            meth_idx = m_idx+2    
            
            if method == 'square_envelope_snr':
                mfp_dist_final = mfp[phases][meth_idx]

    # at the moment only return envelope snr
    return [mfp[phases][0],mfp[phases][1],mfp_dist_final]
        

