import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.tri as tri
from matplotlib import colors
import cmasher as cmr
from pandas import read_csv

#cmap = cmr.seismic    # CMasher
cmap = plt.get_cmap('seismic')   # MPL



def plot_grid(args,grid,data=None,output_file=None,triangulate=False,cbar=False,only_ocean=False,title=None,stationlist_path=None):
    """
    Script to plot a grid with [[lat],[lon]]
    If data is included, should be [[lat],[lon],[data]]
    """
    
    # if no data is included, shouldn't triangulate
    if data is None:
        triangulate = False
    else:
        vmax = np.max(data)
        vmin = -np.max(data)
        
    if stationlist_path is not None:
        stationlist = read_csv(stationlist_path,keep_default_na = False)
        lat = stationlist['lat']
        lon = stationlist['lon']

        if args.correlation_path is not None:

            stat_dict = dict()
            for i in stationlist.iterrows():
                net_1,sta_1,lat_1,lon_1 = i[1]['net'],i[1]['sta'],i[1]['lat'],i[1]['lon']
                stat_dict.update({f"{net_1}.{sta_1}":[lat_1,lon_1]})
            
            # plot only correlation stations
            corr_files = [file for file in os.listdir(args.correlation_path) if file.split(".")[-1] in args.corr_format]

            stat_pair_dict = dict()

            for corr in corr_files:
                net_1 = corr.split('--')[0].split('.')[0]
                sta_1 = corr.split('--')[0].split('.')[1]
                net_2 = corr.split('--')[1].split('.')[0]
                sta_2 = corr.split('--')[1].split('.')[1]

                stat_pair_dict.update({f"{net_1}.{sta_1}":[]})
                stat_pair_dict.update({f"{net_2}.{sta_2}":[]})

            # create lat lon for only correlation pairs
            lat = [stat_dict[i][0] for i in list(stat_pair_dict.keys())]
            lon = [stat_dict[i][1] for i in list(stat_pair_dict.keys())]
            
    
    plt.figure(figsize=(50,20))
    ax  = plt.axes(projection=ccrs.Robinson())
    
    ax.set_global()
    
    if only_ocean:
        ax.add_feature(cfeature.NaturalEarthFeature('cultural', 'admin_0_countries', '50m', edgecolor='black', facecolor=cfeature.COLORS['land']),zorder=2)
    else:
        ax.add_feature(cfeature.COASTLINE,edgecolor='k',linewidth=3)
        ax.add_feature(cfeature.COASTLINE,edgecolor='w',linewidth=1)
        
    if triangulate:
        grid_tri = tri.Triangulation(grid[1],grid[0])
        plt.tripcolor(grid_tri,data,cmap=cmap,vmin=vmin,vmax=vmax,linewidth=0.0,edgecolor='none',zorder=1,transform=ccrs.PlateCarree())

    else:
        # plot either data or grid
        if data is not None:
            plt.scatter(grid[1],grid[0],c=data,s=20,marker='o',cmap=cmap,vmin=vmin,vmax=vmax,zorder=1,transform=ccrs.PlateCarree())
        else:
            plt.scatter(grid[1],grid[0],c='k',s=20,marker='o',cmap=cmap,zorder=1,transform=ccrs.PlateCarree())

    # colorbar
    if cbar:
        cbar = plt.colorbar(pad=0.01)
        cbar.ax.tick_params(labelsize=25)
    
        
    if stationlist_path is not None:
        plt.scatter(lon,lat,c='lawngreen',s=200,marker='^',
                    edgecolor='k',linewidth=2,transform=ccrs.PlateCarree(),zorder=3)
    

    # title
    if title is not None:
        plt.title(title,fontsize=30,pad=25)
    
    # save file? Only plot it if output_file is None
    if output_file is not None:
        plt.savefig(output_file,bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        
    return 
    
