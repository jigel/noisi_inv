"""
Sourcegrid computation

:copyright:
    noisi development team
:license:
    GNU Lesser General Public License, Version 3 and later
    (https://www.gnu.org/copyleft/lesser.html)
"""

import numpy as np
import yaml
import os
import io
from noisi.util.geo import points_on_ell
from noisi.util.source_grid_svp import svp_grid
from noisi.util.source_grid_svp import spherical_distance_degrees
from pandas import read_csv

try:
    from noisi.util.plot import plot_sourcegrid
except ImportError:
    pass
try:
    import cartopy.crs as ccrs
except ImportError:
    pass
import pprint

import functools
print = functools.partial(print, flush=True)


def create_sourcegrid(config,stationlist_path=None):

    if config['verbose']:
        print("Configuration used to set up source grid:", end="\n")
        pp = pprint.PrettyPrinter()
        pp.pprint(config)
        
    # pick either automatic grid, svp grid or normal grid
    if 'auto_data_grid' in config and config["auto_data_grid"]:
                
        from noisi.util.auto_data_grid import setup_data_svpgrid
        
        if config["auto_back_grid_centre"] == 'stations':
            stationlist_path = os.path.join(config["project_path"],'stationlist.csv')
        else:
            stationlist_path = None
        
        
        sigma,beta,phi_min,phi_max,lat_0,lon_0,gamma = setup_data_svpgrid(data_path=config["auto_data_path"],
                             data_thresh=config["auto_data_thresh"],
                             gamma_thresh=config["auto_gamma_thresh"],
                             back_grid_phi_min=config["auto_back_grid_phi_min"],
                             back_grid_phi_max=config["auto_back_grid_phi_max"],
                             data_grid_phi=config["auto_data_grid_phi"],
                             back_grid_centre =config["auto_back_grid_centre"],
                             stationlist_path=stationlist_path,
                             extent = config["auto_extent"], 
                             plot=False)
                
        grid = svp_grid(sigma=sigma,
                          beta=beta,
                          phi_min=phi_min,
                          phi_max=phi_max,
                          lat_0=lat_0,
                          lon_0=lon_0,
                          gamma=gamma,
                          plot=False,
                          dense_antipole=config['svp_dense_antipole'],
                          only_ocean=config['svp_only_ocean'])
        
        # compute and save voronoi cell surface area if set to true

        from noisi.util.geo import get_voronoi_surface_area
            
        grd,surf_areas = get_voronoi_surface_area(grid,config,sigma,beta,phi_min,phi_max,lat_0,lon_0,gamma,output="project")
            
        # reassign grid to make sure it's in the right order
        grid = grd
        
        if config['auto_station_remove'] is not None:
        
            print(f'Removing gridpoints in {config["auto_station_remove"]} radius of stations..')
            
            stationlist = read_csv(stationlist_path)
            lat = stationlist['lat']
            lon = stationlist['lon']
            
            grid_true = np.ones(np.size(grid[0]),dtype=bool)
            grid = np.asarray(grid)


            for i,j in zip(lat,lon):   
                for k,(grd_lon,grd_lat) in enumerate(zip(grid[0],grid[1])):
                    dist_var = spherical_distance_degrees(i,j,grd_lat,grd_lon)        
                    if dist_var < config['auto_station_remove']:
                        grid_true[k] = False    

            grid = grid.T[grid_true].T
            surf_areas = surf_areas[grid_true]
            
        np.save(os.path.join(config['project_path'],'sourcegrid_voronoi.npy'),[grid[0],grid[1],surf_areas])
        print('Voronoi Areas saved as sourcegrid_voronoi.npy')

    
    elif 'svp_grid' in config and config['svp_grid']:
        
        sigma,beta,phi_min,phi_max,lat_0,lon_0,gamma = config['svp_sigma'],config['svp_beta'],config['svp_phi_min'],config['svp_phi_max'],config['svp_lat_0'],config['svp_lon_0'],config['svp_gamma']
        
        grid = svp_grid(sigma=sigma,
                          beta=beta,
                          phi_min=phi_min,
                          phi_max=phi_max,
                          lat_0=lat_0,
                          lon_0=lon_0,
                          gamma=gamma,
                          plot=config['svp_plot'],
                          dense_antipole=config['svp_dense_antipole'],
                          only_ocean=config['svp_only_ocean'])
        
        
        # compute and save voronoi cell surface area if set to true
        if 'svp_voronoi_area' in config and  config['svp_voronoi_area']:
            from noisi.util.geo import get_voronoi_surface_area
            
            grd,surf_areas = get_voronoi_surface_area(grid,config,sigma,beta,phi_min,phi_max,lat_0,lon_0,gamma,output="project")
            
            # reassign grid to make sure it's in the right order
            grid = grd
            
        
        if 'svp_station_remove' in config and config['svp_station_remove'] is not None and stationlist_path is not None:
            
            print(f'Removing gridpoints in {config["svp_station_remove"]} radius of stations..')
            
            stationlist = read_csv(stationlist_path)
            lat = stationlist['lat']
            lon = stationlist['lon']            

            grid_true = np.ones(np.size(grid[0]),dtype=bool)
            grid = np.asarray(grid)


            for i,j in zip(lat,lon):   
                for k,(grd_lon,grd_lat) in enumerate(zip(grid[0],grid[1])):
                    dist_var = spherical_distance_degrees(i,j,grd_lat,grd_lon)        
                    if dist_var < config['svp_station_remove']:
                        grid_true[k] = False    

            grid = grid.T[grid_true].T
            surf_areas = surf_areas[grid_true]            
            
        if config['svp_voronoi_area']:
            np.save(os.path.join(config['project_path'],'sourcegrid_voronoi.npy'),[grid[0],grid[1],surf_areas])
            print('Voronoi Areas saved as sourcegrid_voronoi.npy')
                
            
            
    else:
        grid = points_on_ell(config['grid_dx_in_m'],
                             xmin=config['grid_lon_min'],
                             xmax=config['grid_lon_max'],
                             ymin=config['grid_lat_min'],
                             ymax=config['grid_lat_max'])
    
    sources = np.zeros((2, len(grid[0])))
    sources[0:2, :] = grid

    #if config['verbose']:
    print('Number of gridpoints: ', sources.shape[-1])

    return sources



def setup_sourcegrid(args, comm, size, rank):
    configfile = os.path.join(args.project_path, 'config.yml')
    with io.open(configfile, 'r') as fh:
        config = yaml.safe_load(fh)

    grid_filename = os.path.join(config['project_path'], 'sourcegrid.npy')
    sourcegrid = create_sourcegrid(config)

    # plot
    try:
        plot_sourcegrid(sourcegrid,
                        outfile=os.path.join(config['project_path'],
                                             'sourcegrid.png'),
                        proj=ccrs.PlateCarree)
    except NameError:
        pass

    # write to .npy
    np.save(grid_filename, sourcegrid)

    return()
