# # Gradient-based iterative Inversion
import yaml
import os
import pprint
import shutil
import time
from pandas import read_csv
import numpy as np
import h5py
import sys
from glob import glob
import matplotlib
from obspy import UTCDateTime
matplotlib.use('agg')

from noisi.util.setup_new import setup_proj
from noisi.scripts.source_grid import setup_sourcegrid
from noisi.scripts.run_sourcesetup import source_setup
from noisi.scripts.correlation import run_corr
from noisi.util.corr_obs_copy import copy_corr
from noisi.scripts.run_measurement import run_measurement
from noisi.scripts.kernel import run_kern
from noisi.scripts.assemble_gradient import assemble_grad
from noisi.util.smoothing import smooth
from noisi.util.inv_step_test import steplengthtest
from noisi.util.output_copy import output_copy
from noisi.util.add_metadata import assign_geographic_metadata
from noisi.util.corr_add_noise import corr_add_noise
from noisi.util.output_plot import output_plot
#from noisi.util.obspy_data_download import download_data_inv
from noisi.util.obspy_mass_download import obspy_mass_downloader
from noisi.util.ants_crosscorrelate import ants_preprocess,ants_crosscorrelate
from noisi.scripts.run_wavefieldprep import precomp_wavefield

def precompute_wavefield(args, comm, size, rank):
    pw = precomp_wavefield(args, comm, size, rank)
    pw.precompute()

import functools
print = functools.partial(print, flush=True)


pp = pprint.PrettyPrinter()


from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

########################################################################
# argument class
########################################################################
class args(object):
    def __init__(self):
        pass
        
        
inv_args = args()
start_iter = 0


t_0 = time.time()

# change output folder
try:
    os.chdir(noisi_v2_path)
except: 
    pass


########################################################################
# read in inversion_config.yml and load attributes
########################################################################

with open(sys.argv[1]) as f:
    inv_config = yaml.safe_load(f)
    
# set attributes for inv_args (ignore source_setup_parameters)
for conf in ['main', 'data_download', 'inversion_config', 'project_config', 'grid_config', 'svp_grid_config', 'auto_data_grid_config', 'wavefield_config', 'source_config', 'measr_config']:
    for key in inv_config[conf]:
        setattr(inv_args,key,inv_config[conf][key]) 
        
config = inv_config["project_config"]

if inv_args.svp_grid:
    only_ocean = inv_args.svp_only_ocean
else:
    only_ocean = False
    
    
    
noisi_v2_path = os.getcwd()
if rank == 0:
    print(f"Current directory: {noisi_v2_path}")

    
noisi_path = os.path.join(noisi_v2_path,'noisi')
setattr(inv_args,'noisi_path',noisi_path)

output_path = inv_args.output_folder

if rank == 0:
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

comm.barrier()

os.chdir(output_path)

comm.barrier()

########################################################################
# change to output directory and create project folder
########################################################################

if rank == 0:
    print(f"Changing directory to {os.getcwd()}")

    print("Setting up project..")
    setup_proj(inv_args,comm,size,rank)


comm.barrier()

# read in config.yml file
configfile_proj = f"./{inv_args.project_name}/config.yml"
with open(configfile_proj,"r") as f:
    config_proj = yaml.safe_load(f)

    
setattr(inv_args,'project_path',config_proj["project_path"])

comm.barrier()

if rank == 0:

    # copy inversion_config.yml to project
    inv_conf_path = os.path.join(noisi_v2_path,sys.argv[1])
    print(f"Copying {inv_conf_path} to project folder")
    src = inv_conf_path
    dst = os.path.join(config_proj["project_path"],"inversion_config.yml")
    shutil.copy(src,dst)
    
    # try to copy batch file
    for file in glob(os.path.join(noisi_v2_path,"run_inversion_batch.sbatch")):
        shutil.copy2(file,config_proj["project_path"])
    
    # update project config file
    config_proj.update(inv_config['auto_data_grid_config'])
    config_proj.update(inv_config['svp_grid_config'])
    config_proj.update(inv_config['wavefield_config'])
    config_proj.update(inv_config['grid_config'])
    config_proj.update(inv_config['project_config'])

    with open(configfile_proj,"w") as f:
        yaml.safe_dump(config_proj,f,sort_keys=False)

    #print("Project config file:")
    #pp.pprint(config_proj)
        
comm.barrier()

# time for project
run_time = open(os.path.join(inv_args.project_path,'runtime.txt'),'w+')
run_time.write(f"Number of cores: {size} \n")
t_1 = time.time()
run_time.write(f"Project setup: {np.around((t_1-t_0)/60,4)} \n")

########################################################################
# download (on only one rank) and process data 
########################################################################

if inv_args.download_data:
    
    inv_args.add_metadata = False
    
    # choose date for which data should be downloaded
    if inv_args.download_data_date == "yesterday":
        t_start = UTCDateTime(date.today())-60*60*24
        t_end = UTCDateTime(date.today())
    else:
        t_start = UTCDateTime(inv_args.download_data_date)
        t_end = t_start + 60*60*24
    
    inv_args.t_start = t_start
    inv_args.t_end = t_end
    
    if rank == 0:
        print("="*25)
        print("Downloading and processing data...")
        print("="*25)

        stationlist_new = obspy_mass_downloader(inv_args)

        # change stationlist to new stationlist
        print(f"Copying new stationlist {stationlist_new} to project folder")
        src = stationlist_new
        dst = os.path.join(config_proj["project_path"],"stationlist.csv")
        inv_args.stationlist = dst
        shutil.copy(src,dst)

    
    comm.barrier()

    # time for data download
    t_100 = time.time()
    run_time.write(f"Data Download: {np.around((t_100-t_1)/60,4)} \n")

    comm.barrier()
    
    ants_preprocess(inv_args,comm,size,rank)
    
    comm.barrier()

    # time for data preprocessing
    t_101 = time.time()
    run_time.write(f"Data Pre-processing: {np.around((t_101-t_100)/60,4)} \n")
    
    ants_crosscorrelate(inv_args,comm,size,rank)
    
    comm.barrier()
    
    # time for data cross-correlating
    t_102 = time.time()
    run_time.write(f"Data Cross-correlating: {np.around((t_102-t_101)/60,4)} \n")
    
    # change path to observed cross-correlations
    inv_args.observed_corr = os.path.join(inv_args.project_path,'data','correlations')
    inv_config['inversion_config']['observed_corr'] = inv_args.observed_corr
    
    comm.barrier()
    
    t_1 = time.time()

    if rank == 0:
        print("="*25)
        print("Downloading and processing data done.")
        print("="*25)
        
    
else:
    if rank == 0:
        # copy stationlist
        print(f"Copying {inv_args.stationlist} to project folder")
        src = inv_args.stationlist
        dst = os.path.join(config_proj["project_path"],"stationlist.csv")
        
        inv_args.stationlist = dst
        
        shutil.copy(src,dst)

        
comm.barrier()
    
########################################################################
# setup sourcegrid
########################################################################

if rank == 0:
    print("Setting up sourcegrid..")
    # create the source grid
    setup_sourcegrid(inv_args, comm, size, rank)

comm.barrier()
# time for sourcegrid
t_2 = time.time()
run_time.write(f"Sourcegrid: {np.around((t_2-t_1)/60,4)} \n")

########################################################################
# Convert wavefield
########################################################################


if rank == 0:
    print("Converting wavefield..")
    
if inv_args.wavefield_type == 'greens':
    # copy greens folder to project
    wf_path = os.path.join(inv_args.project_path, 'greens')
    
    if rank == 0:
        print(f"Copying already prepared wavefield from {inv_config['wavefield_config']['wavefield_path']}")
        if os.path.exists(wf_path):
            shutil.rmtree(wf_path)
        shutil.copytree(inv_args.wavefield_path,wf_path)
        print("Copying done.")
    comm.barrier()

    # should check here if wavefield are the same
else:
    precompute_wavefield(inv_args, comm, size, rank)
    #pass
    
comm.barrier()
# time for wavefield
t_3 = time.time()
run_time.write(f"Wavefield: {np.around((t_3-t_2)/60,4)} \n")


########################################################################
# Setup source directory
########################################################################

### change model observed only = True
source_name = 'source_1'
    
setattr(inv_args,'new_model',True)
inv_args.source_model = os.path.join(inv_args.project_path,source_name)
    
if rank == 0:

    print("Creating source directory..")
    # setup source and change config
    # create the source grid
    

    source_setup(inv_args,comm,size,rank)
    
source_configfile = os.path.join(inv_args.project_path,source_name,'source_config.yml')
measr_configfile = os.path.join(inv_args.project_path,source_name,'measr_config.yml')
source_setup_configfile = os.path.join(inv_args.project_path,source_name,'source_setup_parameters.yml')

comm.barrier()

if rank == 0:
    with open(source_configfile) as f:
        config_source = yaml.safe_load(f)

    with open(measr_configfile) as f:
        config_measr = yaml.safe_load(f)
        
    config_measr.update(inv_config['measr_config'])
    config_source.update(inv_config['source_config'])
    config_source.update({'source_setup_file':source_setup_configfile})
    config_sourcesetup = inv_config['source_setup_config']
    
    if inv_args.observed_corr == None:
        config_source['model_observed_only'] = False

    if inv_args.verbose:
        print("Source config files:")
        pp.pprint(config_source)
        pp.pprint(config_sourcesetup)

        print("Measurment config file:")
        pp.pprint(config_measr)
    
    with open(source_configfile,"w") as f:
        yaml.safe_dump(config_source,f,sort_keys=False)

    with open(source_setup_configfile,"w") as f:
        yaml.safe_dump(config_sourcesetup,f,sort_keys=False)
        
    with open(measr_configfile,"w") as f:
        yaml.safe_dump(config_measr,f,sort_keys=False)
        
comm.barrier()

setattr(inv_args,'new_model',False)


########################################################################
# Setup source distribution
########################################################################

# initial source distribution has to be given in yaml file
if rank == 0:        
    print("Setting up source distribution..")
    source_setup(inv_args,comm,size,rank)
    
comm.barrier()

# time for source setup
t_4 = time.time()
run_time.write(f"Source setup: {np.around((t_4-t_3)/60,4)} \n")



########################################################################
# Copy observed cross-correlations
########################################################################

if not os.path.isdir(os.path.join(inv_args.source_model,"observed_correlations_slt")):
    if rank == 0:
        os.makedirs(os.path.join(inv_args.source_model,"observed_correlations_slt"))

if not inv_args.observed_corr == None:
    if rank == 0:
        print("Copying observed correlations")

    n_corr_copy = copy_corr(inv_args,comm,size,rank)

    if n_corr_copy == 0:
        if rank == 0:
            print("Did not copy any observed cross-correlations. Exiting..")

        sys.exit()

comm.barrier()

# time for corr copy
t_5 = time.time()
run_time.write(f"Copy correlations: {np.around((t_5-t_4)/60,4)} \n")

########################################################################
# Begin inversion, first check for already computed iterations
########################################################################

# check for already calculated iterations
models = glob(os.path.join(inv_args.source_model,'iteration*.h5'))
steps = [int(os.path.basename(file).split("_")[1][:-3]) for file in models]

# set attribute
setattr(inv_args,'steplengthrun',False)


if steps != []:
    
    if np.max(steps) == inv_args.nr_iterations:
        if rank == 0:
            print("All iterations already calculated. Exiting..")
        sys.exit()
        
    else:
        if rank == 0:
            print("\n")
            print(f"---------- STARTING AT ITERATION {np.max(steps)} ----------")
            print("\n")

        mf_dict = dict()
        
        setattr(inv_args,'step',np.max(steps))
        start_iter = np.max(steps)
        t_9 = time.time()
        
        # Compute misfit
        measr_step_var = read_csv(os.path.join(inv_args.source_model,f'iteration_{inv_args.step}',f'{inv_args.mtype}.0.measurement.csv'))

        l2_norm_all = np.asarray(measr_step_var['l2_norm'])
        l2_norm = l2_norm_all[~np.isnan(l2_norm_all)]
        mf_step_var = np.mean(l2_norm)

        mf_dict.update({f'iteration_{inv_args.step}':mf_step_var})
        
        if rank == 0:
            print('Misfit dictionary: ',mf_dict)      
else:
    
    ########################################################################
    # Compute iteration 0. Exits if only forward simulation wanted
    ########################################################################
    setattr(inv_args,'step',0)
    
    
    if rank == 0:
        print("\n")
        print(f"---------- ITERATION 0 ----------")
        print("\n")


    if rank == 0:
        print("Computing cross-correlations..")
        
    run_corr(inv_args,comm,size,rank)
    
    comm.barrier()
    #time for corr
    t_6 = time.time()
    run_time.write(f"Correlations iteration_0: {np.around((t_6-t_5)/60,4)} \n")

    if inv_args.add_metadata:
        if rank == 0:
            print("Adding metadata to correlations..")

        corr_path = os.path.join(inv_args.source_model,"iteration_0","corr")
        stationlist_path = inv_args.stationlist
        assign_geographic_metadata(corr_path,stationlist_path,comm,size,rank)

    comm.barrier()
    
    if inv_args.add_noise:
        if rank == 0:
            print("Adding noise to cross-correlations..")
        
        corr_path = os.path.join(inv_args.source_model,"iteration_0","corr")
        corr_add_noise(inv_args,comm,size,rank,corr_path)
            
    comm.barrier()
    

    ########################################################################
    # Correlations
    ########################################################################
    
    if inv_args.observed_corr == None:
        if rank == 0:
            print("Cross-correlations done.")
            print("No observed correlations, exiting..")
        sys.exit()

    ########################################################################
    # Measurement
    ########################################################################
        
    # take measurements 
    if rank == 0:
        print("Cross-correlations done.")
        print("Taking measurement..")

    run_measurement(inv_args,comm,size,rank)

    comm.barrier()
    # time for measurement
    t_7 = time.time()
    run_time.write(f"Measurement iteration_0: {np.around((t_7-t_6)/60,4)} \n")
    
    # check if there are adjoint sources. If not exit since kernels no kernels will be computed
    adjt_path = os.path.join(inv_args.source_model,'iteration_0/adjt')
    if os.listdir(adjt_path) == []:
        if rank == 0:
            print("!"*30)
            print("No adjoint sources found. Can't compute any sensitivity kernels.")
            print("Exiting..")
            
        sys.exit()

    ########################################################################
    # Kernels
    ########################################################################
        
    # compute kernels
    if rank == 0:
        print("Measurements taken.")
        print("Computing sensitivity kernels..")

    run_kern(inv_args,comm,size,rank)

    comm.barrier()
    # time for kern
    t_8 = time.time()
    run_time.write(f"Kernels iteration_0: {np.around((t_8-t_7)/60,4)} \n")

    
    ########################################################################
    # Gradient
    ########################################################################
    
    if rank == 0:
        print("Sensitivity kernels done.")
        print("Assembling gradient..")

    assemble_grad(inv_args,comm,size,rank)

    comm.barrier()

    # time for gradient
    t_9 = time.time()
    run_time.write(f"Gradient iteration_0: {np.around((t_9-t_8)/60,4)} \n")

    if rank == 0:
        print("Gradient assembled.")


    if inv_args.nr_iterations == 0:
        if rank == 0:
            print("0 iterations, exiting..")
        sys.exit()


    # Compute initial misfit
    ########### initial misfit #########
    mf_dict = dict()


    measr_step_var = read_csv(os.path.join(inv_args.source_model,f'iteration_0',f'{inv_config["measr_config"]["mtype"]}.0.measurement.csv'))

    l2_norm_all = np.asarray(measr_step_var['l2_norm'])
    l2_norm = l2_norm_all[~np.isnan(l2_norm_all)]
    mf_step_0 = np.mean(l2_norm)

    mf_dict.update({f'iteration_0':mf_step_0})


    if rank == 0:
        print(f'Misfit for iteration 0: ',mf_step_0)
        print('Misfit dictionary: ',mf_dict)        

    comm.barrier()
    
########################################################################
# Begin inversion loop
########################################################################
    
    
for iter_nr in range(start_iter, inv_args.nr_iterations):

    setattr(inv_args,'step',iter_nr)

    t_9900 = time.time()
    
    # get source distribution and gradient
    source_distr_path = os.path.join(inv_args.source_model,f'iteration_{iter_nr}/starting_model.h5')
    sourcegrid_path=os.path.join(inv_args.project_path,"sourcegrid.npy")

    source_distr = h5py.File(source_distr_path,'r')
    source_grid = np.asarray(source_distr['coordinates'])

    source_distr_data = np.asarray(source_distr['model'])
    source_distr_max = np.max(source_distr_data)
    source_distr_data_norm = source_distr_data/np.max(abs(source_distr_data))

    gradient_path = os.path.join(inv_args.source_model,f'iteration_{iter_nr}','grad','grad_all.npy')
    grad_data = np.load(gradient_path)
    grad_data_norm = grad_data[0]/np.max(abs(grad_data[0]))

    grid_full = source_grid


    ########################################################################
    # Smoothing and step length test
    ########################################################################
    
    # smooth the gradient using noisi function
    grad_smooth_path = os.path.join(inv_args.source_model,f'iteration_{iter_nr}','grad','grad_all_smooth.npy')

    smooth_arr = inv_args.step_smooth
    step = np.asarray([i[0] for i in smooth_arr])
    smooth_var = np.asarray([i[1] for i in smooth_arr])

    step_bool = iter_nr < step

    for j in range(np.size(step_bool)):
        if step_bool[j]:
            sigma_degrees = smooth_var[j]
            break
        elif not any(step_bool):
            sigma_degrees = smooth_var[-1]
            break
        else:
            continue    

    sigma=[sigma_degrees*111000]
    
    if rank == 0:
        print(f'Smoothing iteration {iter_nr} with {sigma_degrees} degree smoothing...')
        np.save(os.path.join(inv_args.source_model,f'iteration_{iter_nr}','grad',f'smoothing_iter_{iter_nr}.npy'),sigma_degrees)

    comm.barrier()

    smooth(gradient_path,grad_smooth_path,sourcegrid_path,sigma=sigma,cap=95,thresh=1e-10,comm=comm,size=size,rank=rank)
    
    comm.barrier()

    t_9901 = time.time()
    run_time.write(f"Smoothing iteration_{inv_args.step}: {np.around((t_9901-t_9900)/60,4)} \n")
    
    if rank == 0:
        print(f'Smoothing iteration {iter_nr} done.')
        print("Performing step length test..")
        

    # step length test
    step_length,slt_success = steplengthtest(inv_args,comm,size,rank,mf_dict,gradient_path,grad_smooth_path,sourcegrid_path,sigma=sigma,cap=95,thresh=1e-10)

    
    comm.barrier()
    
    if not slt_success:
        if rank == 0:
            print('!'*60)
            print(f'!!! Misfit cannot be reduced for iteration {inv_args.step} !!!')
            print(f'!!!!!! Stopping Inversion !!!!!')
            print('!'*60)        
            
        break
    
    
    ########################################################################
    # Update model
    ########################################################################

    t_9902 = time.time()
    run_time.write(f"Steplengthtest iteration_{inv_args.step}: {np.around((t_9902-t_9901)/60,4)} \n")
 
    inv_args.step += 1
    setattr(inv_args,'steplengthrun',False)

    # update noise distribution
    if rank == 0:
        print("Updating noise source distribution..")
        
        grad_smooth = np.load(grad_smooth_path).T
        
        # take minimum value and normalise 
        #grad_smooth_norm = grad_smooth/np.abs(np.min(grad_smooth))
        grad_smooth_norm = grad_smooth/np.max(np.abs(grad_smooth))

        distr_update_norm = source_distr_data_norm - grad_smooth_norm*step_length

        # set all negative values to 0
        distr_update_norm[distr_update_norm < 0] = 0
        
        # scale back up again 
        distr_update = np.asarray([distr_update_norm*source_distr_max][0])
        
        # square the distribution then normalise and rescale
        #distr_update_2 = np.power(distr_update,2)
        #distr_update_2_norm = distr_update_2/np.max(np.abs(distr_update_2))
        #distr_update = distr_update_2_norm * source_distr_max
        

        # save new distribution in new .h5 file
        sourcemodel_new = os.path.join(inv_args.source_model,f'iteration_{inv_args.step}.h5')
        shutil.copy2(source_distr_path,sourcemodel_new)
            
        
        with h5py.File(sourcemodel_new) as fh:
            del fh['model']
            fh.create_dataset('model',data=distr_update.astype(np.float32))
                
        print("\n")
        print(f"---------- ITERATION {iter_nr+1} ----------")
        print("\n")
                
        print(f"New sourcemodel for iteration {inv_args.step}: ", os.path.join(inv_args.source_model,f'iteration_{inv_args.step}.h5')) 

        # make step folder
        if not os.path.exists(os.path.join(inv_args.source_model,f'iteration_{inv_args.step}')):
            os.makedirs(os.path.join(inv_args.source_model,f'iteration_{inv_args.step}'))

            for d in ['adjt','grad','corr','kern']:
                os.mkdir(os.path.join(inv_args.source_model,f'iteration_{inv_args.step}',d))

        # copy sourcemodel to starting_model.h5 and base_model.h5
        shutil.copy2(sourcemodel_new,os.path.join(inv_args.source_model,f'iteration_{inv_args.step}','starting_model.h5'))

        print(f'Made iteration_{inv_args.step} folder.')
        
    comm.barrier()

    t_9903 = time.time()
    run_time.write(f"Noisesource Update iteration_{inv_args.step}: {np.around((t_9903-t_9902)/60,4)} \n")
    
    ########################################################################
    # Correlations
    ########################################################################
    
    if rank == 0:
        print(f"Computing cross-correlations for iteration {inv_args.step}..")
    
    run_corr(inv_args,comm,size,rank)
    
    comm.barrier()
   
    t_9904 = time.time()
    run_time.write(f"Correlations iteration_{inv_args.step}: {np.around((t_9904-t_9903)/60,4)} \n")

    if inv_args.add_noise:
        if rank == 0:
            print("Adding noise to cross-correlations..")
            
        corr_path = os.path.join(inv_args.source_model,"iteration_0","corr")
        corr_add_noise(inv_args,comm,size,rank,corr_path)
            
    comm.barrier()
    
    ########################################################################
    # Measurement
    ########################################################################
 
    if rank == 0:
        print(f"Cross-correlations for iteration {inv_args.step} done.")
        print(f"Taking measurement for iteration {inv_args.step}..")
        
    run_measurement(inv_args,comm,size,rank)

    comm.barrier()
   
    t_9905 = time.time()
    run_time.write(f"Measurement iteration_{inv_args.step}: {np.around((t_9905-t_9904)/60,4)} \n")

    ########################################################################
    # Kernel
    ########################################################################
    
    if rank == 0:
        print(f"Measurements for iteration {inv_args.step} taken.")
        print(f"Computing sensitivity kernels for iteration {inv_args.step}..")

    run_kern(inv_args,comm,size,rank)
    
    comm.barrier()

    t_9906 = time.time()
    run_time.write(f"Kernels iteration_{inv_args.step}: {np.around((t_9906-t_9905)/60,4)} \n")   
 
    ########################################################################
    # Gradient 
    ########################################################################.
    
    if rank == 0:
        print(f"Sensitivity kernels for iteration {inv_args.step} done.")
        print(f"Assembling gradient for iteration {inv_args.step}..")
    
    assemble_grad(inv_args,comm,size,rank)
    
    comm.barrier()

    t_9907 = time.time()
    run_time.write(f"Gradient iteration_{inv_args.step}: {np.around((t_9907-t_9906)/60,4)} \n")

    
    # Compute misfit
    measr_step_var = read_csv(os.path.join(inv_args.source_model,f'iteration_{inv_args.step}',f'{inv_config["measr_config"]["mtype"]}.0.measurement.csv'))

    l2_norm_all = np.asarray(measr_step_var['l2_norm'])
    l2_norm = l2_norm_all[~np.isnan(l2_norm_all)]
    mf_step_var = np.mean(l2_norm)

    mf_dict.update({f'iteration_{inv_args.step}':mf_step_var})


    if rank == 0:
        print(f'Misfit for iteration {inv_args.step}: ',mf_step_var)
        print('Misfit dictionary: ',mf_dict)        
        
        
if rank == 0:
    print(f"Inversion done. Misfit reduced from {np.around(list(mf_dict.values())[0],2)} to {np.around(list(mf_dict.values())[-1],2)}")

# inversion time
t_10 = time.time()
run_time.write(f"Inversion: {np.around((t_10-t_9)/60,4)} \n")
run_time.write(f"Total runtime: {np.around((t_10-t_0)/60,4)} \n")
run_time.close()

if rank == 0:
    output_copy(inv_args.project_path)
    print("Files copied to output folder.")

    
comm.barrier()

if rank == 0:
    if inv_args.output_plot:
        output_plot(os.path.join(inv_args.project_path,"output"),only_ocean=only_ocean,triangulation=True)

        
comm.barrier()

if rank == 0:
    print("================ INVERSION SCRIPT DONE ===================")

