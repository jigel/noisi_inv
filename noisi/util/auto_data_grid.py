# automatically setup spatially variable grid

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy as cart
from noisi.util.source_grid_svp import spherical_distance_degrees
#from obspy.signal.invsim import cosine_taper
import instaseis
import os
import json
import sys
from pandas import read_csv

def setup_data_svpgrid(data_path,data_thresh=0.1,gamma_thresh=5,back_grid_phi_min=0.8,back_grid_phi_max=2,data_grid_phi=0.5,back_grid_centre = 'data',stationlist_path=None,extent = None, plot=False):
    """
    This function automatically sets up the spatially variable grid
    
    Input: 
    data_path :: .npy file with lat, lon, data.
    
    data_thresh :: dense grid will be added for points above this threshold. Data is normalised so 1 is max. 
    gamma_thresh :: radius in which points will be removed. (e.g. 10)
    back_grid_centre :: One of 'data', 'stations', [longitude, latitude]. If 'stations', need stationlist_path 
    back_grid_phi :: background gridpoint distance in degrees (e.g. 4)
    data_grid_phi :: additional data gridpoint distance in degrees (e.g. 0.5)
    
    extent :: [lon_min,lon_max,lat_min,lat_max] gives area in which grid will be made denser
    plot :: True/False
    
    Output:
    sigma,beta,phi_min,phi_max,lat_0,lon_0,gamma 
    """
    
    grid_data_norm = np.load(data_path)
    
    lat_ini = []
    lon_ini = []
    data_ini = []
    
    
    # iterate over all grid points
    # data is in grid_data
    
    for i in range(0,np.size(grid_data_norm[2])):
        if grid_data_norm[2][i] >= data_thresh:
            lat_ini.append(grid_data_norm[0][i])
            lon_ini.append(grid_data_norm[1][i])
            data_ini.append(grid_data_norm[2][i])
        else:
            continue
            
            
    lat_ini = np.asarray(lat_ini)
    lon_ini = np.asarray(lon_ini)
    data_ini = np.asarray(data_ini)
    print("Number of gridpoints above threshold: ", np.size(lat_ini))
    
    
    ## Remove all points above threshold that are not within area
    if extent is not None:
        lon_min = extent[0]
        lon_max = extent[1]
        lat_min = extent[2]
        lat_max = extent[3]
        
        lat_ini_cut = []
        lon_ini_cut = []
        data_ini_cut = []

        for i in range(0,np.size(lat_ini)):
            if lat_min < lat_ini[i] < lat_max and lon_min < lon_ini[i] < lon_max:
                lat_ini_cut.append(lat_ini[i])
                lon_ini_cut.append(lon_ini[i])
                data_ini_cut.append(data_ini[i])
            else:
                continue
                
        lat_ini = lat_ini_cut
        lon_ini = lon_ini_cut
        data_ini = data_ini_cut

        
    # Now remove grids that are not really necessary
    
    lat_new = []
    lon_new = []
    data_new = []
    
    lat_var = []
    lon_var = []
    data_var = []
    
    # Sort them in reverse so that strongest point definitely is one centre
    data_ini,lat_ini,lon_ini = zip(*sorted(zip(data_ini,lat_ini,lon_ini),reverse=True))


    
    j = 0
    
    while j < np.size(lat_ini):
        if j == 0:
            for i in range(0,np.size(lat_ini)):
                lat_var.append(lat_ini[0])
                lon_var.append(lon_ini[0])
                data_var.append(data_ini[0])
                dist_var = spherical_distance_degrees(lat_ini[0],lon_ini[0],lat_ini[i],lon_ini[i])
                if dist_var > gamma_thresh:
                    lat_var.append(lat_ini[i])
                    lon_var.append(lon_ini[i])
                    data_var.append(data_ini[i])
                else:
                    continue
        else:
            try:
                lat_new = lat_var
                lon_new = lon_var
                data_new = data_var
                
                lat_var = []
                lon_var = []
                data_var = []
                
                lat_var.append(lat_new[j])
                lon_var.append(lon_new[j])
                data_var.append(data_new[j])
                for i in range(0,np.size(lat_new)):
                    dist_var = spherical_distance_degrees(lat_new[j],lon_new[j],lat_new[i],lon_new[i])
                    if dist_var > gamma_thresh:
                        lat_var.append(lat_new[i])
                        lon_var.append(lon_new[i])
                        data_var.append(data_new[i])
                    else:
                        continue
            except:
                break          
        j += 1
        
                
             
    print("Number of additional grids:", np.size(lat_new))

        
    # BACKGROUND GRID: SVP
    
    # compute geographical centre of additional grid centres
    if back_grid_centre == 'data' or back_grid_centre is None:
        lat_back = np.sum(lat_new)/np.size(lat_new)
        lon_back = np.sum(lon_new)/np.size(lon_new)
        
        
        # compute sigma for background grid:
        # additional grid with centre furthest away from back grid centre should give us sigma
        dist_back_grid = []

        for i in range(0,np.size(lat_new)):
            dist_var = spherical_distance_degrees(lat_back,lon_back,lat_new[i],lon_new[i])
            dist_back_grid.append(dist_var)

        back_grid_sigma = np.max(dist_back_grid)
        
    elif back_grid_centre == 'stations':
        # read stationlist file
        stationlist = read_csv(stationlist_path)
        lat_stations = np.asarray(stationlist['lat'])
        lon_stations = np.asarray(stationlist['lon'])
        
        lat_back = np.sum(lat_stations)/np.size(lat_stations)
        lon_back = np.sum(lon_stations)/np.size(lon_stations)
        
        # compute sigma for background grid:
        # sigma here should be distance between station centre and furthest grid point above threshold
        dist_back_grid = []        

        for i in range(0,np.size(lat_new)):
            dist_var = spherical_distance_degrees(lat_back,lon_back,lat_new[i],lon_new[i])
            dist_back_grid.append(dist_var)

        back_grid_sigma = np.max(dist_back_grid)
        
        
    elif np.size(back_grid_centre)==2:
        lat_back = back_grid_centre[0]
        lon_back = back_grid_centre[1]
        
        # compute sigma for background grid:
        # additional grid with centre furthest away from back grid centre should give us sigma
        dist_back_grid = []        

        for i in range(0,np.size(lat_new)):
            dist_var = spherical_distance_degrees(lat_back,lon_back,lat_new[i],lon_new[i])
            dist_back_grid.append(dist_var)

        back_grid_sigma = np.max(dist_back_grid)

        
    
    # Now create variables for svp grids
    
    # Background grid
    sigma = [back_grid_sigma]
    beta = [30]
    phi_min = [back_grid_phi_min]
    phi_max = [back_grid_phi_max]
    lat_0 = [lat_back]
    lon_0 = [lon_back]
    #n = [200]
    gamma = [0]
    
    sigma_var = []
    beta_var = []
    phi_min_var = []
    phi_max_var = []
    lat_0_var = []
    lon_0_var = []
    gamma_var = []
    #plot = False
    #dense_antipole = False
    #only_ocean = True
    
    for i in range(0,np.size(lat_new)):
        lat_0_var.append(lat_new[i])
        lon_0_var.append(lon_new[i])
        sigma_var.append(20)
        beta_var.append(5)
        
        # variable grid density
        # approaches data_grid_phi+back_grid_phi_min)/2 as the value of data gets smaller
        if data_grid_phi > back_grid_phi_min:
            print('data_grid_phi has to be less than back_grid_phi_min')
            print('Exiting. Please change in config file.')
            sys.exit()
        if data_grid_phi == back_grid_phi_min:
            phi_data = data_grid_phi
        else:
            phi_data = data_grid_phi + (1-(data_new[i]/np.max(data_new)))*(((back_grid_phi_min-data_grid_phi)/2))
    
        phi_min_var.append(phi_data)
        phi_max_var.append(phi_data)
        gamma_var.append((data_new[i]+1)*gamma_thresh)
        
    # sorts them so that the densest grid is added last
    phi_min_var,sigma_var,beta_var,phi_max_var,lat_0_var,lon_0_var,gamma_var = zip(*sorted(zip(phi_min_var,sigma_var,beta_var,phi_max_var,lat_0_var,lon_0_var,gamma_var),reverse=True))
    
    sigma += sigma_var
    beta += beta_var
    phi_min += phi_min_var
    phi_max += phi_max_var
    lat_0 += lat_0_var
    lon_0 += lon_0_var
    gamma += gamma_var
    
    return sigma,beta,phi_min,phi_max,lat_0,lon_0,gamma
