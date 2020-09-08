import numpy as np
import os
import yaml
import h5py
from noisi.scripts.correlation import run_corr
from noisi.scripts.run_measurement import run_measurement
from glob import glob
from shutil import copy2
from shutil import move
from shutil import rmtree
from pandas import read_csv
from noisi.util.smoothing import smooth
from noisi.util.corr_add_noise import corr_add_noise

import functools
print = functools.partial(print, flush=True)

def steplengthtest(args,comm,size,rank,mf_dict,gradient_path,grad_smooth_path,sourcegrid_path,sigma,cap=95,thresh=1e-10):
    """
    Performs a steplengthtest for the inversion.
    
    args: project_path, source_model, step
    output: step_length
    """
    
    # load input
    source_distr_path = os.path.join(args.source_model,f'iteration_{args.step}/starting_model.h5')

    source_distr = h5py.File(source_distr_path)
    source_grid = np.asarray(source_distr['coordinates'])

    source_distr_data = np.asarray(source_distr['model'])
    source_distr_max = np.max(source_distr_data)
    
    if np.all(source_distr_data==0):
        source_distr_data_norm = source_distr_data
    else:
        source_distr_data_norm = source_distr_data/np.max(abs(source_distr_data))

    grid_full = source_grid
    
    
    # could also put these in config
    n_step_tests = args.nr_step_tests
    step_length_test = np.linspace(args.step_length_min,args.step_length_max,args.nr_step_tests)

    if rank == 0:
        try:
            os.makedirs(os.path.join(args.source_model,f'iteration_{args.step}','step_length_tests'))
            print(f'Made step_length_tests directory for iteration_{args.step}.')
        except:
            pass

    comm.barrier()
    
    #misfit_step_test = [[0,list(mf_dict.values())[-1]]]
    misfit_step_test = []
    slt_step_count = 9900
    iter_count = 0
    iter_actual = args.step
    slt_success = False
    
    while True:
        
        args.step = iter_actual

        grad_smooth_path = os.path.join(args.source_model,f'iteration_{args.step}','grad','grad_all_smooth.npy')
        grad_smooth = np.load(grad_smooth_path).T

        # take minimum value and normalise 
        #grad_smooth_norm = grad_smooth/np.abs(np.min(grad_smooth))
        grad_smooth_norm = grad_smooth/np.max(np.abs(grad_smooth))

        step_test = step_length_test[iter_count]
                
        # Update noise source distribution
        distr_update_norm = source_distr_data_norm - grad_smooth_norm*step_test
            
        # set all negative values to 0
        distr_update_norm[distr_update_norm < 0] = 0
        # scale back up again 
        
        if source_distr_max == 0:
            distr_update = np.asarray([distr_update_norm][0])
        else:
            distr_update = np.asarray([distr_update_norm*source_distr_max][0])
            
        if not any(distr_update):
            if rank == 0 :
                print("Distribution is 0 everywhere after update. Check gradient.")
            break
            #raise ValueError("Distribution is 0 everywhere.")
            
        if rank == 0:
            # save new distribution in new .h5 file
            sourcemodel_new = os.path.join(args.source_model,f'iteration_{args.step}','step_length_tests',f'sourcemodel_slt_{args.step}_{slt_step_count}.h5')
            copy2(source_distr_path,sourcemodel_new)    
    
            with h5py.File(sourcemodel_new) as fh:
                del fh['model']
                fh.create_dataset('model',data=distr_update.astype(np.float32))
                fh.close() 
                

            # make step folder
            if not os.path.exists(os.path.join(args.source_model,f'iteration_{slt_step_count}')):
                os.makedirs(os.path.join(args.source_model,f'iteration_{slt_step_count}'))

                for d in ['corr','adjt']:
                    os.mkdir(os.path.join(args.source_model,f'iteration_{slt_step_count}',d))
                    
                    
            # copy sourcemodel to starting_model.h5
            copy2(sourcemodel_new,os.path.join(args.source_model,f'iteration_{slt_step_count}','starting_model.h5'))
    
    
        comm.barrier()
        
        
        if rank == 0:
            print(f"Computing correlations for step length test {slt_step_count}...")

        # change args.step        
        args.step = slt_step_count
        args.steplengthrun = True
        args.verbose = False

        run_corr(args,comm,size,rank)
        
        comm.barrier()
        
        if args.add_noise:
            if rank == 0:
                print("Adding noise to cross-correlations..")
            
            corr_path = os.path.join(args.source_model,"iteration_0","corr")
            corr_add_noise(args,comm,size,rank,corr_path)

        comm.barrier()

        if rank == 0:
            print(f'Computing measurements for step length test {slt_step_count}...')   

        run_measurement(args,comm,size,rank)
        
        comm.barrier()

        ### get misfit

        measr_iter_var = read_csv(glob(os.path.join(args.source_model,f'iteration_{slt_step_count}',"*measurement.csv"))[0])

        l2_norm_all = np.asarray(measr_iter_var['l2_norm'])
        
        l2_norm = l2_norm_all[~np.isnan(l2_norm_all)]
        mf_step_var = np.mean(l2_norm)

        if rank == 0:
            #print('Misfit step test: ',misfit_step_test)
            print(f'Step length {np.around(step_test,3)} has misfit {np.around(mf_step_var,6)}.')
            
        
        # test if it's misfit increase or decrease
        if args.step_test_smoothing and iter_count == 0 and mf_step_var > list(mf_dict.values())[-1]:
            if rank == 0:
                print('/'*40)
                print('/// No reduction in misfit ///')
                print('/'*40)

            sigma_old = sigma
            sigma = np.asarray(sigma) - 0.5*111000
            
            if sigma == 0:
                if rank ==0:
                    print('!'*40)
                    print('!!! Misfit cannot be reduced !!!')
                    print('!'*40)
                    
                slt_success = False
                break
                
            else:
                if rank == 0:
                    print(f"Reducing smoothing from {sigma_old[0]/111000} to {sigma[0]/111000}.")
                

                smooth(gradient_path,grad_smooth_path,sourcegrid_path,sigma=sigma,cap=95,thresh=1e-10,comm=comm,size=size,rank=rank)

                if rank == 0:
                    for folder in os.listdir(args.source_model):
                        if folder.startswith('iteration_99'):
                            rmtree(os.path.join(args.source_model,folder))
                            
            comm.barrier()
                        
        else:
            
            misfit_step_test.append([step_test,mf_step_var])
            slt_success = True
            slt_step_count += 1
            iter_count += 1
            
            if iter_count == np.size(step_length_test):
                break

            
    comm.barrier()
    
    args.step = iter_actual

            
    if slt_success:
        # new step length by fitting polynomial on rank 0 

        # get new step length by fitting polynomial
        sl = [i[0] for i in misfit_step_test]
        mf = [i[1] for i in misfit_step_test]

        p = np.polyfit(sl,mf,2)

        x = np.linspace(-np.max(sl) * 2, np.max(sl) * 2.5, 5000)
        y = p[0] * x ** 2 + p[1] * x + p[2]

        misfit_step_test_fit = [x,y]

        # make sure it doesn't do weird stuff
        # only positive step lengths permitted
        # step length not larger than two times maximum step length
        if x[np.argmin(y)] > np.min(mf):
            step_length = sl[np.argmin(mf)]
            mf_pred = np.min(mf)
        elif p[0] < 0:
            step_length = sl[np.argmin(mf)]
            mf_pred = np.min(mf)
        elif x[np.argmin(y)] < 0:
            step_length = sl[np.argmin(mf)]
            mf_pred = np.min(mf)
        elif x[np.argmin(y)] > 2*args.step_length_max:
            step_length = 2*args.step_length_max
            mf_pred = p[0] * step_length ** 2 + p[1] * step_length + p[2]
        else:
            step_length = x[np.argmin(y)]
            mf_pred = np.min(y)

        if rank == 0:
            print('Misfit step tests: ',misfit_step_test)
            print(f'Optimal step length: {np.around(step_length,2)}')
            print(f'Misfit value predicted at optimal step length: {np.around(mf_pred,2)}')

            misfit_step_test = [misfit_step_test,[step_length,mf_pred]]

            # save misfit_step_test array
            np.save(os.path.join(args.source_model,f'iteration_{args.step}','misfit_step_test.npy'),misfit_step_test)
            np.save(os.path.join(args.source_model,f'iteration_{args.step}','misfit_step_test_fit.npy'),misfit_step_test_fit)
            
                        
    if rank == 0:
        print("Deleting step length test iterations..")
        for folder in os.listdir(args.source_model):
            if folder.startswith('iteration_99'):
                rmtree(os.path.join(args.source_model,folder))

    comm.barrier()
    
    if slt_success:
        return step_length,slt_success
    
    else:
        return 0,slt_success
    
    
