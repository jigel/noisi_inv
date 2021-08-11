"""
Plot essential files in the output folder

:copyright:
    noisi development team
:license:
    GNU Lesser General Public License, Version 3 and later
    (https://www.gnu.org/copyleft/lesser.html)
"""


import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy as cart
import cartopy.feature as cfeature
import matplotlib.tri as tri
from matplotlib import colors
import os
from glob import glob
from pandas import read_csv
import h5py
from obspy import UTCDateTime
from datetime import date

import functools
print = functools.partial(print, flush=True)

try:
    import cmasher as cmr
    cmap = cmr.cosmic    # CMasher
    cmap = plt.get_cmap('cmr.cosmic')   # MPL
    cmash_import = True
except:
    cmash_import = False
    

def output_plot(args,output_path,only_ocean=False,triangulation=False):
    """
    Plot files in output folder
    """
    print("Plotting..")

    stationlist_path = glob(os.path.join(output_path,'stationlist*'))[0]
    #print(stationlist_path)
    station_weight = False

    try:
        sourcegrid_path = glob(os.path.join(output_path,'sourcegrid.npy'))[0]
    except:
        print('Need sourcegrid.')

    #print(sourcegrid_path)

    stationlist = read_csv(stationlist_path,keep_default_na=False)
    lat = list(stationlist['lat'])
    lon = list(stationlist['lon'])

    # make plot folder
    if not os.path.exists(os.path.join(output_path,'output_plots')):
        os.makedirs(os.path.join(output_path,'output_plots'))

    output_plots = os.path.join(output_path,'output_plots')


    # plot misfit
    # misfit
    steps_avail_path = [os.path.join(output_path,i) for i in os.listdir(output_path) if i.startswith('iteration') and not os.path.isfile(os.path.join(output_path,i))]

    misfit_step = []
    misfit_dict = dict()

    for j in steps_avail_path:

        measr_file_paths_var = [os.path.join(j,i) for i in os.listdir(j) if i.endswith('measurement.csv')]

        #print(measr_file_paths_var)
        if measr_file_paths_var == []:
            print(f'No measurement for {os.path.basename(j)}')

        else:
            i = measr_file_paths_var[0]
            step_nr_var = int(i.split('/')[-2].split('_')[1])
            measr_step_var = read_csv(i,keep_default_na=False)

            l2_norm_all = np.asarray(measr_step_var['l2_norm'])
            l2_norm = l2_norm_all[~np.isnan(l2_norm_all)]
            mf_step_var = np.mean(l2_norm)

            misfit_step.append([step_nr_var,mf_step_var])  
            misfit_dict.update({step_nr_var:mf_step_var})


    misfit = [i[1] for i in misfit_step]
    step = [i[0] for i in misfit_step]

    step,misfit = zip(*sorted(zip(step,misfit)))

    mf_reduc = (1-misfit[-1]/misfit[0])*100

    #### MISFIT PLOTS
    plt.figure(figsize=(15,8))
    plt.plot(step,misfit,'k',marker='o',markerfacecolor='b',markersize=10)
    plt.grid()
    plt.xlabel('Iterations',fontsize=15)
    plt.ylabel('Misfit',fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title(f'Misfit reduced by {np.around(mf_reduc,2)}%',pad=20,fontsize=30)
    plt.savefig(os.path.join(output_plots,'misfit_vs_iterations.png'),bbox_inches='tight',facecolor='white',dpi=72)
    #plt.show()
    plt.close()


    ## plot steplength tests
    steplength_files = [os.path.join(i,'misfit_step_test.npy') for i in steps_avail_path if os.path.isfile(os.path.join(i,'misfit_step_test.npy'))]
    steplength_fit_files = [os.path.join(i,'misfit_step_test_fit.npy') for i in steps_avail_path if os.path.isfile(os.path.join(i,'misfit_step_test_fit.npy'))]

    print("Plotting steplength tests..")

    for i in range(np.size(steplength_files)):

        step = steplength_files[i].split('/')[-2].split('_')[1]

        mf_step = np.asarray(np.load(steplength_files[i],allow_pickle=True)[0])
        mf_final_step = np.load(steplength_files[i],allow_pickle=True)[1]
        mf_step_fit = np.load(steplength_fit_files[i],allow_pickle=True)

        step_m = [i[0] for i in mf_step]
        mf_m = [i[1] for i in mf_step]

        plt.figure(figsize=(15,8))
        plt.scatter(step_m,mf_m,c='r')
        #plt.scatter(mf_step[:,0],mf_step[:,1],c='r')
        plt.scatter(mf_final_step[0],mf_final_step[1],c='r',s=100,marker='x',label='Predicted misfit')
        plt.scatter(mf_final_step[0],misfit_dict[int(step)+1],c='b',s=100,marker='x',label='Actual misfit')
        plt.plot(mf_step_fit[0],mf_step_fit[1],c='b')
        plt.grid()
        plt.xlabel("Steplength",fontsize=15)
        plt.ylabel("Misfit",fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.title(f'Steplength test for iteration {step} with final step {np.around(mf_final_step[0],2)}. Predicted misfit: {np.around(mf_final_step[1],2)}. Actual misfit: {np.around(misfit_dict[int(step)+1],2)}',pad=30,fontsize=20)
        plt.legend()
        plt.savefig(os.path.join(output_plots,f'iteration_{step}_0_slt.png'),bbox_inches='tight',facecolor='white',dpi=72)
        #plt.show()
        plt.close()

        
    # plot distributions
    #print(os.listdir(source_path))
    source_path = glob(os.path.join(output_path,'models'))[0]

    #sourcemodel_paths = [os.path.join(source_path,i) for i in os.listdir(source_path) if i.startswith('iteration')]
    sourcemodel_paths = glob(os.path.join(source_path,"iteration*.h5"))
    #print(sourcemodel_paths)
    sourcemodel_steps = []

    for model in sourcemodel_paths:
        step_var = os.path.basename(model).split('_')[-1].split('.')[0] 
        sourcemodel_steps.append(int(step_var))

    # sort them
    sourcemodel_steps,sourcemodel_paths = zip(*sorted(zip(sourcemodel_steps,sourcemodel_paths)))


    ### plot gradient and smoothed gradient

    grad_paths = []
    grad_smooth_paths = []
    grad_smoothing_paths = []

    for path in steps_avail_path:
        grad_paths.append(os.path.join(path,'grad_all.npy'))
        grad_smooth_paths.append(glob(os.path.join(path,'grad_all_smooth.npy')))
        grad_smoothing_paths.append(glob(os.path.join(path,'smoothing*')))

    grad_smoothing_dict = dict()

    for file in grad_smoothing_paths:
        try:
            step = file[0].split('/')[-2].split('_')[1]
            smooth_var = float(np.load(file[0],allow_pickle=True))
            grad_smoothing_dict.update({step:smooth_var})
        except:
            #print('fail')
            pass


    print("Plotting sourcemodels..")
        
    for model in sourcemodel_paths:
        step = os.path.basename(model).split('_')[-1].split('.')[0]    
        if step == 'sourcemodel':
            step = 0

        #print(step)
        source_distr_file = h5py.File(model,'r')
        source_grid = np.asarray(source_distr_file['coordinates'])
        source_distr = np.asarray(source_distr_file['model']).T[0]
        source_distr_norm = source_distr/np.max(np.abs(source_distr))

        plt.figure(figsize=(50,20))
        ax = plt.axes(projection=ccrs.Robinson(central_longitude=0))
        ax.set_global()

        if only_ocean:
            ax.add_feature(cfeature.NaturalEarthFeature('cultural', 'admin_0_countries', '50m', edgecolor='black', facecolor=cfeature.COLORS['land']),zorder=2)
        else:
            ax.coastlines()
        
        if triangulation:
            triangles = tri.Triangulation(source_grid[0],source_grid[1])
            
            if cmash_import:
                plt.tripcolor(triangles,source_distr_norm,cmap=cmap,linewidth=0.0,edgecolor='none',vmin=0,vmax=1,zorder=1,transform=ccrs.Geodetic())
            else:
                plt.tripcolor(triangles,source_distr_norm,cmap=plt.get_cmap('Blues_r'),linewidth=0.0,edgecolor='none',vmin=0,vmax=1,zorder=1,transform=ccrs.Geodetic())

        else:
            
            if cmash_import:
                plt.scatter(source_grid[0],source_grid[1],s=20,c=source_distr_norm,vmin=0,vmax=1,transform=ccrs.PlateCarree(),cmap=cmap,zorder=3)
            else:
                plt.scatter(source_grid[0],source_grid[1],s=20,c=source_distr_norm,vmin=0,transform=ccrs.PlateCarree(),cmap=plt.get_cmap('Blues_r'),zorder=3)
                

        cbar = plt.colorbar(pad=0.01)
        cbar.ax.tick_params(labelsize=30) 
        cbar.set_label('Normalised Power Spectral Density',rotation=90,labelpad=50,fontsize=50)
        
        
        if args.download_data:
                        # choose date for which data should be downloaded
            if args.download_data_date == "yesterday":
                t_inv = UTCDateTime(date.today())-60*60*24*args.download_data_days
            else:
                t_inv = UTCDateTime(args.download_data_date)
            
            
            plt.title(f"Inversion for {str(t_inv).split('T')[0]}: iteration {step}",pad=30,fontsize=50)

        else:
            plt.title(f'Noise distribution for iteration {step}',pad=30,fontsize=50)
        #try:
        #    plt.title(f'Noise distribution for iteration {step} with misfit {np.around(misfit_dict[int(step)],2)}',fontsize=50)
        #except:
        #    plt.title(f'Noise distribution for iteration {step}',fontsize=50)

        plt.scatter(lon,lat,s=150,c='lawngreen',marker='^',edgecolor='k',linewidth=1,transform=ccrs.PlateCarree(),zorder=4)
        plt.savefig(os.path.join(output_plots,f'iteration_{step}_1_noise_distribution.png'),bbox_inches='tight',facecolor='white',dpi=50)
        #plt.show()
        plt.close()


    print("Plotting gradients..")
        
    grid = np.load(sourcegrid_path)
    if triangulation:
        triangles = tri.Triangulation(grid[0],grid[1])

    for file in grad_paths:
        try:
            grad = np.load(file,allow_pickle=True)[0]
            v = np.max(np.abs(grad))

            step = file.split('/')[-2].split('_')[1]

            plt.figure(figsize=(50,20))
            ax = plt.axes(projection=ccrs.Robinson(central_longitude=0))
            ax.set_global()

            if only_ocean:
                ax.add_feature(cfeature.NaturalEarthFeature('cultural', 'admin_0_countries', '50m', edgecolor='black', facecolor=cfeature.COLORS['land']),zorder=2)
            else:
                ax.coastlines()
            
            if triangulation:
                plt.tripcolor(triangles,grad,cmap=plt.get_cmap('seismic'),linewidth=0.0,edgecolor='none',vmin=-v,vmax=v,zorder=1,transform=ccrs.Geodetic())
            else:
                plt.scatter(grid[0],grid[1],s=20,c=grad,transform=ccrs.Geodetic(),cmap='seismic',vmin=-v,vmax=v,zorder=3)

            cbar = plt.colorbar(pad=0.01)
            cbar.formatter.set_powerlimits((0, 0))
            cbar.ax.tick_params(labelsize=30) 
            cbar.set_label('Power Spectral Density',rotation=270,labelpad=40,fontsize=40)

            try:
                plt.title(f'Gradient for iteration {step} with misfit {np.around(misfit_dict[int(step)],2)}',pad=30,fontsize=50)
            except:
                plt.title(f'Gradient for iteration {step}',pad=30,fontsize=50)
                         
            plt.scatter(lon,lat,s=150,c='lawngreen',marker='^',edgecolor='k',linewidth=1,transform=ccrs.PlateCarree(),zorder=3)
            plt.savefig(os.path.join(output_plots,f'iteration_{step}_2_gradient.png'),bbox_inches='tight',facecolor='white',dpi=50)
            #plt.show() 
            plt.close()
            
        except:
            pass


    for file in grad_smooth_paths:
        if file == []:
            continue
        else:
            grad = np.load(file[0],allow_pickle=True)[0]
            v = np.max(np.abs(grad))

            step = file[0].split('/')[-2].split('_')[1]

            plt.figure(figsize=(50,20))
            ax = plt.axes(projection=ccrs.Robinson(central_longitude=0))
            ax.set_global()

            if only_ocean:
                ax.add_feature(cfeature.NaturalEarthFeature('cultural', 'admin_0_countries', '50m', edgecolor='black', facecolor=cfeature.COLORS['land']),zorder=2)
            else:
                ax.coastlines()
            
            if triangulation:
                plt.tripcolor(triangles,grad,cmap=plt.get_cmap('seismic'),linewidth=0.0,edgecolor='none',vmin=-v,vmax=v,zorder=1,transform=ccrs.Geodetic())
            else:
                plt.scatter(grid[0],grid[1],s=20,c=grad,transform=ccrs.Geodetic(),cmap='seismic',vmin=-v,vmax=v,zorder=3)

            cbar = plt.colorbar(pad=0.01)
            cbar.formatter.set_powerlimits((0, 0))
            cbar.ax.tick_params(labelsize=30) 
            cbar.set_label('Power Spectral Density',rotation=270,labelpad=40,fontsize=40)

            try:
                plt.title(f'Smoothed gradient for iteration {step} with misfit {np.around(misfit_dict[int(step)],2)} and {grad_smoothing_dict[str(step)]}° smoothing',pad=30,fontsize=50)
            except:
                plt.title(f'Smoothed gradient for iteration {step}',pad=30,fontsize=50
                         )
            plt.scatter(lon,lat,s=150,c='lawngreen',marker='^',edgecolor='k',linewidth=1,transform=ccrs.PlateCarree(),zorder=3)
            plt.savefig(os.path.join(output_plots,f'iteration_{step}_3_gradient_smooth.png'),bbox_inches='tight',facecolor='white',dpi=50)
            #plt.show()
            plt.close()

    print("Plotting station sensitivity..")    
        
    # station sensitivity
    try:
        stat_sensitivity_path = os.path.join(output_path,'station_sensitivity.npy')
        stat_sensitivity = np.load(stat_sensitivity_path)

        stat_sensitivity_norm = stat_sensitivity/np.max(np.abs(stat_sensitivity))

        v = 0.2

        plt.figure(figsize=(50,20))
        ax = plt.axes(projection=ccrs.Robinson(central_longitude=0))
        ax.set_global()

        if only_ocean:
            ax.add_feature(cfeature.NaturalEarthFeature('cultural', 'admin_0_countries', '50m', edgecolor='black', facecolor=cfeature.COLORS['land']),zorder=2)
        else:
            ax.coastlines()
        
        #if triangulation:
        #    plt.tripcolor(triangles,stat_sensitivity_norm,cmap=plt.get_cmap('RdBu_r'),linewidth=0.0,edgecolor='none',vmax=v,zorder=1,transform=ccrs.Geodetic())
        #else:
        #    plt.scatter(grid[0],grid[1],s=20,c=stat_sensitivity_norm,transform=ccrs.Geodetic(),cmap='RdBu_r',vmax=v,zorder=3)
        
        if triangulation:
            triangles = tri.Triangulation(source_grid[0],source_grid[1])
            
            if cmash_import:
                plt.tripcolor(triangles,stat_sensitivity_norm,cmap=cmap,linewidth=0.0,edgecolor='none',vmin=0,vmax=v,zorder=1,transform=ccrs.Geodetic())
            else:
                plt.tripcolor(triangles,stat_sensitivity_norm,cmap=plt.get_cmap('Blues_r'),linewidth=0.0,edgecolor='none',vmin=0,vmax=v,zorder=1,transform=ccrs.Geodetic())

        else:
            
            if cmash_import:
                plt.scatter(source_grid[0],source_grid[1],s=20,c=stat_sensitivity_norm,vmin=0,vmax=v,transform=ccrs.PlateCarree(),cmap=cmap,zorder=3)
            else:
                plt.scatter(source_grid[0],source_grid[1],s=20,c=stat_sensitivity_norm,vmin=0,vmax=v,transform=ccrs.PlateCarree(),cmap=plt.get_cmap('Blues_r'),zorder=3)
                

            
            
        cbar = plt.colorbar(pad=0.01)
        cbar.ax.tick_params(labelsize=30) 
        cbar.set_label('Normalised Sensitivity',rotation=270,labelpad=40,fontsize=40)

        #cbar.set_label('Power Spectral Density',rotation=270,labelpad=10)
        plt.title(f'Station Sensitivity with vmax = {v}',pad=30,fontsize=50)
        plt.scatter(lon,lat,s=150,c='lawngreen',marker='^',edgecolor='k',linewidth=1,transform=ccrs.PlateCarree(),zorder=3)
        plt.savefig(os.path.join(output_plots,f'station_sensitivity.png'),bbox_inches='tight',facecolor='white',dpi=50)
        #plt.show()
        plt.close()
        
        
        plt.figure(figsize=(50,20))
        ax = plt.axes(projection=ccrs.Robinson(central_longitude=0))
        ax.set_global()

        ax.add_feature(cfeature.NaturalEarthFeature('cultural', 'admin_0_countries', '50m', edgecolor='black', facecolor=cfeature.COLORS['land']),zorder=2)
        plt.scatter(grid[0],grid[1],s=20,c='k',transform=ccrs.PlateCarree(),zorder=3)
        #cbar.set_label('Power Spectral Density',rotation=270,labelpad=10)
        plt.title(f'Sourcegrid',pad=30,fontsize=50)
        plt.scatter(lon,lat,s=150,c='lawngreen',marker='^',edgecolor='k',linewidth=1,transform=ccrs.PlateCarree(),zorder=3)
        plt.savefig(os.path.join(output_plots,f'sourcegrid.png'),bbox_inches='tight',facecolor='white',dpi=50)
        #plt.show()
        plt.close()
        
        
        # plot final model with sensitivity below 1% set to 0
        #sense_min = 0.01

        #stat_sensitivity_cap = stat_sensitivity_norm.copy()
        #stat_sensitivity_cap[stat_sensitivity_norm<sense_min] = 0
        #stat_sensitivity_cap[stat_sensitivity_norm>sense_min] = 1

        #model = sourcemodel_paths[-1]
        
        #step = os.path.basename(model).split('_')[-1].split('.')[0]    
        #if step == 'sourcemodel':
        #    step = 0

        #print(step)
        #source_distr_file = h5py.File(model,'r')
        #source_grid = np.asarray(source_distr_file['coordinates'])
        #source_distr = np.asarray(source_distr_file['model']).T[0]
        #source_distr_norm = source_distr/np.max(np.abs(source_distr))

        #plt.figure(figsize=(50,20))
        #ax = plt.axes(projection=ccrs.Robinson(central_longitude=0))
        #ax.set_global()
        
        #if only_ocean:
        #    ax.add_feature(cfeature.NaturalEarthFeature('cultural', 'admin_0_countries', '50m', edgecolor='black', facecolor=cfeature.COLORS['land']),zorder=2)
        #else:
        #    ax.coastlines()
        
        #if triangulation:
        #    triangles = tri.Triangulation(source_grid[0],source_grid[1])
        #    plt.tripcolor(triangles,stat_sensitivity_cap*source_distr_norm,cmap=plt.get_cmap('RdBu_r'),vmin=0,linewidth=0.0,edgecolor='none',zorder=1,transform=ccrs.Geodetic())

        #else:
        #    plt.scatter(source_grid[0],source_grid[1],s=20,c=stat_sensitivity_cap*source_distr_norm,vmin=0,transform=ccrs.Geodetic(),cmap='RdBu_r',zorder=3)

        #cbar = plt.colorbar(pad=0.01)
        #cbar.ax.tick_params(labelsize=30) 
        #cbar.set_label('Power Spectral Density',rotation=270,labelpad=40,fontsize=40)

        #try:
        #    plt.title(f'Noise distribution for iteration {step} capped at {sense_min*100}% sensitivity with misfit {np.around(misfit_dict[int(step)],2)}',fontsize=50)
        #except:
        #    plt.title(f'Noise distribution for iteration {step} capped at {sense_min*100}% sensitivity',fontsize=50)

        #plt.scatter(lon,lat,s=150,c='lawngreen',marker='^',edgecolor='k',linewidth=1,transform=ccrs.PlateCarree(),zorder=4)
        #plt.savefig(os.path.join(output_plots,f'iteration_{step}_2_noise_distribution_capped.png'),bbox_inches='tight',facecolor='white')
        #plt.show()
        #plt.close()
        
        
        
        
    except:
        print("Could not plot station sensitivity.")    
        
        
        
    print("Plotting ray coverage for kernels..")
    
    try:
        kern_path = os.path.join(args.project_path,'source_1/iteration_0/kern')
        kern_files = glob(os.path.join(kern_path,'*.npy'))

        # if kern_files empty, try getting used_obs_corr_list.csv
        if kern_files == []:
            sta_pair_used_file = glob(os.path.join(output_path,'used_obs_corr_list.csv'))

            if sta_pair_used_file != []:

                sta_pair = read_csv(sta_pair_used_file[0],header=None)

                kern_pairs = [f"{i[1][0].split('--')[0].split('.')[0]}.{i[1][0].split('--')[0].split('.')[1]}--{i[1][0].split('--')[1].split('.')[0]}.{i[1][0].split('--')[1].split('.')[1]}" for i in sta_pair.iterrows()]

            else:
                pass

        else:
            kern_pairs = [f"{os.path.basename(i).split('--')[0].split('.')[0]}.{os.path.basename(i).split('--')[0].split('.')[1]}--{os.path.basename(i).split('--')[1].split('.')[0]}.{os.path.basename(i).split('--')[1].split('.')[1]}" for i in kern_files]


        station_dict = dict()
        station_anti_dict = dict()

        for sta_i in stationlist.iterrows():
            sta = sta_i[1]
            station_dict.update({f"{sta['net']}.{sta['sta']}":[sta['lat'],sta['lon']]})

            if sta['lon'] < 0:
                station_anti_dict.update({f"{sta['net']}.{sta['sta']}":[-sta['lat'],sta['lon']+180]})
            elif sta['lon'] >= 0:
                station_anti_dict.update({f"{sta['net']}.{sta['sta']}":[-sta['lat'],sta['lon']-180]})


        kern_stat_dict = {}
        n_rays = 0

        plt.figure(figsize=(50,20))
        ax = plt.axes(projection=ccrs.Robinson())
        ax.add_feature(cfeature.NaturalEarthFeature('cultural', 'admin_0_countries', '50m', edgecolor='black', facecolor=cfeature.COLORS['land']),zorder=1)
        #ax.coastlines()
        ax.set_global()

        for stat_pair in kern_pairs:
            i = stat_pair.split('--')[0]
            j = stat_pair.split('--')[1]


            kern_stat_dict.update({i:[station_dict[i][0],station_dict[i][1]]})
            kern_stat_dict.update({j:[station_dict[j][0],station_dict[j][1]]})

            plt.plot([station_dict[i][1],station_anti_dict[j][1]],[station_dict[i][0],station_anti_dict[j][0]], color='k',zorder=2, transform=ccrs.Geodetic(),alpha=4/np.size(list(station_dict.keys())))
            plt.plot([station_anti_dict[i][1],station_dict[j][1]],[station_anti_dict[i][0],station_dict[j][0]], color='k',zorder=2, transform=ccrs.Geodetic(),alpha=4/np.size(list(station_dict.keys())))

            n_rays += 1

        stat_lat = np.asarray(list(kern_stat_dict.values())).T[0]
        stat_lon = np.asarray(list(kern_stat_dict.values())).T[1]

        plt.scatter(stat_lon,stat_lat,s=150,c='lawngreen',marker='^',edgecolor='k',linewidth=1,transform=ccrs.PlateCarree(),zorder=3)

        plt.title(f"Ray coverage with {n_rays} rays",pad=30,fontsize=50)
        plt.savefig(os.path.join(output_plots,f'kernel_ray_coverage.png'),bbox_inches='tight',facecolor='white',dpi=50)

        plt.close()
            
    except Exception as e:
        print(e)
        print("Could not plot ray coverage.")


    print(f"Plots can be found in {output_plots}")
