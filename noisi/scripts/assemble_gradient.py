"""
Adjoint sources for computing noise source sensitivity
kernels in noisi
:copyright:
    noisi development team
:license:
    GNU Lesser General Public License, Version 3 and later
    (https://www.gnu.org/copyleft/lesser.html)
"""

import numpy as np
import os
from glob import glob
from noisi.my_classes.noisesource import NoiseSource
from noisi.scripts.correlation import config_params

import functools
print = functools.partial(print, flush=True)

def assemble_grad(args, comm, size, rank):
    """
    args: source_model, step
    
    Sums up all kernels to get gradient for inversion
    Only works for one initial distribution at the moment
    """
    
    args.steplengthrun = False
    all_conf = config_params(args, comm, size, rank)

    project_path = all_conf.source_config['project_path']
    source_path = os.path.join(project_path,all_conf.source_config['source_name'])
    
    
    kern_dir = os.path.join(source_path,f"iteration_{all_conf.step}","kern")
    kern_files = glob(os.path.join(kern_dir,"*.npy"))

    nsrc = NoiseSource(os.path.join(source_path,f"iteration_{all_conf.step}","starting_model.h5"))
    grd = np.load(os.path.join(all_conf.source_config['project_path'],"sourcegrid.npy"))

    n_dist = nsrc.spect_basis.shape[0]
    
    gradient = np.zeros((n_dist,np.shape(grd)[1]))
    
    
    # make list of all indices which is then split between cores
    n = np.size(kern_files)
    
    kern_idx = np.arange(0,n,1)
        
    if rank == 0:
        kern_idx_split = np.array_split(kern_idx,size)
        kern_idx_split = [k.tolist() for k in kern_idx_split]
    else:
        kern_idx_split = None
        
    kern_idx_split = comm.scatter(kern_idx_split,root=0)
    
    for i in kern_idx_split:
    
        kern_var = np.sum(np.load(kern_files[i]),axis=2)

        for dist in range(n_dist):
            gradient[dist,:] += kern_var[dist,:]
            
            
    # MPI COLLECT HERE
    gradient_coll = comm.gather(gradient,root=0)
    
    if rank == 0:    
        # add them all up into one
        gradient = np.sum(gradient_coll,axis=0)

        # save kernel file
        grad_file = os.path.join(source_path,f"iteration_{all_conf.step}","grad","grad_all.npy")

        np.save(grad_file,gradient)
        
        print('Gradient computed and saved.')
        
    return gradient

    
    
    
