import numpy as np
from shutil import copy2
import os
from glob import glob
import sys
from pandas import read_csv

import functools
print = functools.partial(print, flush=True)


def output_copy(project_path):

    # project path
    print(project_path)
    
    # create
    if not os.path.exists(os.path.join(project_path,'output')):
        os.makedirs(os.path.join(project_path,'output'))
    
    output_path = os.path.join(project_path,'output')
    
    
    # copy stationlist and inversion config file
    copy2(glob(os.path.join(project_path,'stationlist*'))[0],output_path)
    print(f'Copied {glob(os.path.join(project_path,"stationlist*"))[0]}')
    
    copy2(glob(os.path.join(project_path,'inversion_config*'))[0],output_path)
    print(f'Copied {glob(os.path.join(project_path,"inversion_config*"))[0]}')
    
    copy2(os.path.join(project_path,'sourcegrid.npy'),output_path)
    print(f'Copied {os.path.join(project_path,"sourcegrid.npy")}')
    
    try:
        copy2(os.path.join(project_path,'sourcegrid_voronoi.npy'),output_path)
        print(f'Copied {os.path.join(project_path,"sourcegrid_voronoi.npy")}')
    except:
        pass
    
    copy2(os.path.join(project_path,'runtime.txt'),output_path)
    print(f'Copied {os.path.join(project_path,"runtime.txt")}') 
          
    # copy sourcemodel steps
    if not os.path.exists(os.path.join(output_path,'models')):
        os.makedirs(os.path.join(output_path,'models'))
        
    sourcemodel_path = os.path.join(output_path,'models')
    #print(sourcemodel_path)
    
    source_path = glob(os.path.join(project_path,'source_*'))[0]
    #print(source_path)
    
    for file in glob(os.path.join(source_path,'*.yml')):
        copy2(file,output_path)

    for file in glob(os.path.join(project_path,'*.yml')):
        copy2(file,output_path)  
        
    for file in glob(os.path.join(project_path,'*batch*')):
        copy2(file,output_path)    
    
    
    for file in glob(os.path.join(source_path,'*.h5')):
        copy2(file,sourcemodel_path)
  
    # copy starting model
    start_model = os.path.join(source_path,'iteration_0','starting_model.h5')
    sm_dst = os.path.join(sourcemodel_path,'iteration_0.h5')
    copy2(start_model,sm_dst)
              
              
    # copy measurement and gradient for each step
    for step in glob(os.path.join(source_path,'iteration_*')):
    
        if os.path.isdir(step):
    
            if not os.path.exists(os.path.join(output_path,os.path.basename(step))):
                os.makedirs(os.path.join(output_path,os.path.basename(step)))
            
            step_path = os.path.join(output_path,os.path.basename(step))
        
    	    # copy step test files
            mf_step_files = glob(os.path.join(step,'misfit_step_*'))
            print(f'Copying {mf_step_files} to {step_path}')
            for mf_file in mf_step_files:
                    copy2(mf_file,step_path)
    	        
            # copy measurement file
            measr_file = glob(os.path.join(step,'*measurement*'))
            print(f'Copying {measr_file} to {step_path}')
            for m_file in measr_file:
                copy2(m_file,step_path)
    
            # copy smoothing file
            smooth_file = glob(os.path.join(step,'grad','smoothing*'))
            print(f'Copying {smooth_file} to {step_path}')
            for m_file in smooth_file:
                copy2(m_file,step_path)
    
            # copy gradient 
            grad_file = glob(os.path.join(step,'grad','*grad*'))
            print(f'Copying {grad_file} to {step_path}')
    
            for g_file in grad_file:
                copy2(g_file,step_path)
    
    
    
    
    # need to unweight kernels for station sensitivity    
    print('Computing station sensitivity..')
    kern_path = os.path.join(source_path,'iteration_0','kern')
    measr_0_file = glob(os.path.join(source_path,'iteration_0','*measurement*'))[0]
    print('..loading files and summing..')
    
    measr_data = read_csv(measr_0_file)

    # dictionary with station pairs and synth obs
    measr_dict = dict()

    for i,pair in measr_data.iterrows():
        
        sta1_var = pair['sta1'].split('.')[0] + '.' + pair['sta1'].split('.')[1]
        sta2_var = pair['sta2'].split('.')[0] + '.' + pair['sta2'].split('.')[1]
        
        measr_dict.update({f"{sta1_var}--{sta2_var}":[pair['syn'],pair['obs']]})
        
        
    for i,file in enumerate(glob(os.path.join(kern_path,'*.npy'))):
        kern_var = np.load(file)[0].T[0]
        
        file_name = os.path.basename(file)
        
        kern1_var = file_name.split('.')[0] + '.' + file_name.split('.')[1]
        kern2_var = file_name.split('--')[1].split('.')[0] + '.' + file_name.split('--')[1].split('.')[1]
        
        kern_pair = f"{kern1_var}--{kern2_var}"

        measr_var = measr_dict[kern_pair][0] - measr_dict[kern_pair][1]

        if i == 0:
            stat_sense = np.abs(kern_var/measr_var)

        else:
            stat_sense += np.abs(kern_var/measr_var)

    try: 
        print('..saving..')
        np.save(os.path.join(project_path,'station_sensitivity.npy'),stat_sense)
        print('..copying..')
        copy2(os.path.join(project_path,'station_sensitivity.npy'),output_path)
        print('Station sensitivity computed.')
    except: 
        print('Could not save station sensitivity.')

    return


