import obspy
import numpy as np
from glob import glob
import os 
import yaml
import pandas as pd

def corr_add_noise(args,comm,size,rank,corr_path,perc=0.05):
    
    
    # get bandpass filter to filter noise
    with open(os.path.join(args.source_model,'measr_config.yml')) as mconf:
            measr_config = yaml.safe_load(mconf)
    # Figure out how many frequency bands are measured
    bandpass = measr_config['bandpass']

    if bandpass is None:
        pass
    elif type(bandpass) == list:
        if rank == 0:
            print("Using first bandpass filter in list:", bandpass)
        pass
    else:
        if rank == 0:
            print("Error with bandpass. Noise not filtered.")
        pass

    # initialise for mpi
    corr_rms = 0
    
    #if rank == 0:
    
    # load all correlations to get rms
    corr_files = glob(os.path.join(corr_path,"*"))

    #### split up files so that it can be scattered
    if rank == 0:
        files_split = np.array_split(corr_files,size)
        files_split = [k.tolist() for k in files_split]
    else:
        files_split = None
        
    comm.barrier()
        
    files_split = comm.scatter(files_split,root=0) 
    

    corr_rms_arr = []
    
    # get rms for error
    for file in files_split:

        tr = obspy.read(file)[0]
        data = tr.data
        corr_rms_arr.append(np.sqrt(np.sum(data**2)))
        
        
        
    # collect all corr_rms_arr and take the mean
    corr_rms_all = comm.gather(corr_rms_arr,root=0)
    # send it to all again
    corr_rms_all = comm.bcast(corr_rms_all,root=0)
    # create flat list
    corr_rms_final = [i for l in corr_rms_all for i in l]

    # take mean
    corr_rms = np.mean(corr_rms_final)

    comm.barrier()
        
    # add noise
    for i,file in enumerate(files_split):
        
        if args.verbose:
            if i%100 == 0:
                print(f"At {i} of {np.size(files_split)}")
        
        st = obspy.read(file)

        # add noise to data
        corr_noise_init = np.random.randn(np.shape(st[0].data)[0])
        #normalise noise
        corr_noise = (corr_noise_init/np.max(np.abs(corr_noise_init)))*corr_rms*perc
        
        if bandpass is None:
            st[0].data += corr_noise
            
        elif type(bandpass) == list:

            # filter the noise
            tr_noise = obspy.Trace(data=corr_noise)
            tr_noise.filter(type='bandpass',freqmin=bandpass[0][0],freqmax=bandpass[0][1],corners=bandpass[0][2])
            corr_noise_filt = tr_noise.data

            st[0].data += corr_noise_filt
            
        else:
            st[0].data += corr_noise

        
        st.write(file)

    
    return 
    
    