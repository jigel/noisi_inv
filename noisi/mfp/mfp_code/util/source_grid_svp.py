import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import json
import os
import io
import warnings
warnings.filterwarnings("ignore")

def svp_grid_one(sigma,beta,phi_min,phi_max,lat_0,lon_0,plot=False,dense_antipole = True,only_ocean=False):
    """
    This function creates a Spatially Variable grid. Input parameters:
    :sigma (greater than 2) = standard deviation, i.e. size of the area of denser grid points
    :beta = steepness of the drop of to the maximum distance
    :phi_min = initial distance between grid points, in degrees
    :phi_max = maximum distance between grid points, in degrees
    :lat_0 = latitude of point of interest, -90° to 90°
    :lon_0 = longitude of point of interest, -180° to 180°
    :plot = True/False
    :dense_antipole = True/False
    :only_ocean = True/False
    Returns: list of longitudes and latitudes
    """
    
    # Error messages
    if lat_0 < -90 or lat_0 > 90:
        msg = 'lat_0 has to be between -90° and 90°'
        raise ValueError(msg)
    
    if lon_0 < -180 or lon_0 > 180:
        msg = 'lon_0 has to be between -180 and 180'
        raise ValueError(msg)
        
    if phi_min > 90 or phi_max > 90:
        msg = 'phi_min and phi_max should not be larger than 90'
        raise ValueError(msg)
    
    if phi_min > phi_max:
        msg = 'phi_min should be smaller than phi_max'
        raise ValueError(msg)

    # Step 1: SVP
    # Calculate radii of the circles

    # have a constant value from 0 to sigma degrees with phi_min
    dphi_min = int(sigma/phi_min)*[phi_min]
    # the sum of all shouldn't be bigger than 180
    dphi = dphi_min

    i = 0
    if phi_min==phi_max:
        while np.sum(dphi) < 180:
            #print(np.sum(dphi))
            dphi.append(phi_min)
    else:
        phi_max = phi_max - phi_min

        while np.sum(dphi) < 180:
            #print(np.sum(dphi))
            # gaussian
            dphi1 = phi_max*(1 - np.exp(-(i/10)**beta))+phi_min

            if dphi1 > phi_min+0.01:
                dphi.append(dphi1)
            i += 0.1

    # remove last entry since it's above 180
    dphi1 = dphi[:-1]


    phi = []
    dphi = []
    phi_0 = 0
    
    if dense_antipole:
        for i in range(0,np.size(dphi1)):
            phi_0 += dphi1[i]
            phi.append(phi_0)
            dphi.append(dphi1[i])
            # Change condition so that if the distance between equator and previous circle is greater than that before the point is removed
            if phi_0 > 90:
                if dphi[i] > dphi[i-1]:
                    if 90-phi[i-1] < dphi[i-1]:
                        phi = phi[:-2]  # removes last two entries of phi
                        dphi = dphi[:-2] # removes last two dphi
                        phi_0 = 90
                        phi.append(phi_0)
                        dphi.append(90-phi[i-2])
                        break
                    else:
                        phi = phi[:-1]  # removes last entry of phi since it would be bigger than 90
                        dphi = dphi[:-1] # removes last phi
                        break
                elif dphi[i] <= dphi[i-1]: 
                    if 90-phi[i-1] < dphi[i-1]:
                        phi = phi[:-2]  # removes last two entries of phi
                        dphi = dphi[:-2] # removes last two dphi
                        phi_0 = 90
                        phi.append(phi_0)
                        dphi.append(90-phi[i-2])
                        break
                    else:
                        phi = phi[:-1]  # removes last entry of phi since it would be bigger than 90
                        dphi = dphi[:-1] # removes last phi
                        break       
                else:
                    phi = phi[:-1]  # removes last entry of phi since it would be bigger than 90
                    dphi = dphi[:-1] # removes last phi
                    break
    else:
        for i in range(0,np.size(dphi1)):
            phi_0 += dphi1[i]
            phi.append(phi_0)
            dphi.append(dphi1[i])            
            if phi_0 > 180:
                phi = phi[:-1]  # removes last entry of phi since it would be bigger than 180
                dphi = dphi[:-1] # removes last phi
                break                

    # phi now consists of the latitudinal angles up to 90° over which we should loop

    # Step 2: Longitudinal angles (azmiuth)
    # We want these angles to be close to the latitudinal distance between the circles

    ### OLD BELOW
    #dtheta = dphi
    #N = np.around(np.divide(360,dtheta))
    
    # To get the angle we now use 2*Pi/N
    #theta = np.divide(2*np.pi,N)
    ### OLD ABOVE
    
    #### Try with circumference of circles

    r_earth = 6371 #km

    # Use phi
    S = np.abs(2*np.pi*r_earth*np.sin(np.deg2rad(phi)))

    # Now have circumference in km. One degree = 111km. 
    # To get number of points on each circle we divide S by the distance between the circles dphi
    N = np.around(np.divide(S,np.multiply(dphi,111)))

    theta = np.divide(2*np.pi,N)


    ## We now have the latitudes and the angle over which we have to loop to get the longitudes. 
    # Step 3: Loop

    lat_final1 = [0]
    lon_final1 = [0]
    

    for i in range(0,np.size(phi)-1):
        for j in range(0,int(N[i])):
            # first need to calculate all longitudes
            lon_final1.append(np.rad2deg(j*theta[i]))
            lat_final1.append(phi[i])
            
    # need to shift it
    lon_final1 = np.subtract(lon_final1,180)
    lat_final1 = np.subtract(lat_final1,90)
    
    
    if dense_antipole:
        # Now flip the longitude to make the other hemisphere
        lat_final2 = [0]
        lon_final2 = [0] 

        for i in range(0,np.size(phi)):    # size(phi) - 1 since we don't want two sets of points around the equator
            for j in range(0,int(N[i])):
                # first need to calculate all longitudes
                lon_final2.append(np.rad2deg(j*theta[i]))
                lat_final2.append(phi[i])

        # Shift coordinates and flip it
        lon_final2 = np.subtract(lon_final2,180)
        lat_final2 = np.subtract(lat_final2,90)
        lat_final2 = np.multiply(lat_final2,-1)
        lon_final2 = lon_final2

        # Combine the two
        lat_final = np.concatenate((lat_final1,lat_final2))
        lon_final = np.concatenate((lon_final1,lon_final2))
    else:
        lat_final = lat_final1
        lon_final = lon_final1
    
    # Calculate grid point distance of densest grid area in m for latitude
    dlat = 111132.954 - 559.822 * np.cos(2*lat_0) + 1.175*np.cos(4*lat_0)
    dx_min = dphi[0]*dlat
    dx_max = dphi[-1]*dlat
        
    #print('Number of Gridpoints:',np.size(lat_final))
    if plot:
        print('Minimum dx in m:',np.around(dx_min,3),'m which is',np.around(dphi[0],3),'°')
        print('Maximum dx in m:',np.around(dx_max,3),'m which is',np.around(dphi[-1],3),'°')

    # ROTATION TO AREA OF INTEREST
    # We have the variables lon_final, lat_final
    # lat_0 and lon_0 give the point to which the center should be shifted in degrees

    # Step 1: Convert lat_final & lon_final from degrees to radians
    if dense_antipole:
        lat_final_rad = np.deg2rad(lat_final)
        lon_final_rad = np.deg2rad(lon_final)
    else:
        lat_final_rad = -np.deg2rad(lat_final)
        lon_final_rad = -np.deg2rad(lon_final)

    # Rotation around y-axis and z-axis
    theta_rot = np.deg2rad(90-lat_0)
    phi_rot = np.deg2rad(lon_0)

    # Convert grid from spherical to cartesian 
    x_cart = np.cos(lon_final_rad)*np.cos(lat_final_rad)
    y_cart = np.sin(lon_final_rad)*np.cos(lat_final_rad)
    z_cart = np.sin(lat_final_rad)

    # Shift the coordinates, it's the two rotation matrices for z and y multiplied.
    x_cart_new = np.cos(theta_rot)*np.cos(phi_rot)*x_cart + np.cos(theta_rot)*np.sin(phi_rot)*y_cart + np.sin(theta_rot)*z_cart
    y_cart_new = -np.sin(phi_rot)*x_cart + np.cos(phi_rot)*y_cart
    z_cart_new = -np.sin(theta_rot)*np.cos(phi_rot)*x_cart - np.sin(theta_rot)*np.sin(phi_rot)*y_cart + np.cos(theta_rot)*z_cart

    # Convert cartesian back to spherical coordinates
    # Brute force for longitude because rotation matrix does not seem to rotate the longitudes
    lon_final_rot = []
    
    for i in range(0,np.size(lon_final_rad)):
        lon_final_rot.append(np.rad2deg(np.arctan2(y_cart_new[i],x_cart_new[i])+phi_rot))
        if lon_final_rot[i] > 180:
            lon_final_rot[i] = lon_final_rot[i] - 360
        elif lon_final_rot[i] < -180:   
            lon_final_rot[i] = lon_final_rot[i] + 360
            
    lat_final_rot = np.rad2deg(np.arcsin(z_cart_new))

    
    if only_ocean:
        from noisi.util.geo import is_land
        
        # new function in noisi: 0 if land 1 if ocean
        is_ocean = np.abs(is_land(lon_final_rot,lat_final_rot) - 1.)
        
        grid_onlyocean_lon = []
        grid_onlyocean_lat = []

        for i in range(0,np.size(lat_final_rot)):
            if is_ocean[i] == 1:
                grid_onlyocean_lon.append(lon_final_rot[i])
                grid_onlyocean_lat.append(lat_final_rot[i])
            else:
                continue
        
        lon_final_final = grid_onlyocean_lon
        lat_final_final = grid_onlyocean_lat
    else:
        lon_final_final = lon_final_rot
        lat_final_final = lat_final_rot
        
    #print('Number of gridpoints:',np.size(lat_final_final))
    
    if plot:
        print('Number of gridpoints:',np.size(lat_final_final))
        plt.figure(figsize=(25,10))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.coastlines()
        plt.scatter(lon_final_final,lat_final_final,s=1,c='k',transform=ccrs.PlateCarree())
        plt.title('Centre at %0.2f ° latitude and %0.2f ° longitude with %i gridpoints' %(lat_0,lon_0,np.size(lat_final_final)))
        plt.show(block=False)
        #plt.close()

        
    return list((lon_final_final,lat_final_final))


def spherical_distance_degrees(lat1,lon1,lat2,lon2):
    """
    Calculate ths distance between two points in degrees
    Input: latitude 1, longitude 1, latitude 2, longitude 2
    """
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    hav = np.sin(np.deg2rad(dlat)/2)**2 + np.cos(np.deg2rad(lat1)) * np.cos(np.deg2rad(lat2)) * np.sin(np.deg2rad(dlon)/2)**2
    dist_deg = np.rad2deg(2 * np.arctan2(np.sqrt(hav), np.sqrt(1-hav)))

    return dist_deg


def svp_grid(sigma=[15],beta=[5],phi_min=[1],phi_max=[3],lat_0=[0],lon_0=[0],gamma=[0],plot=False,dense_antipole=False,only_ocean=False):
    """
    This function creates several spatially variable grids. To turn them into one grid the variable gamma has to be defined. 
    The input should be arrays with the variables for the svp grid plus gamma.
    The first grid does not need a gamma, but a random value should be given. This grid will be used as the base grid.
    Example input: [sigma1,sigma2],[beta1,beta2],[phi_min1,phi_min2],....,[gamma1,gamma2],[True,True],[False,True],[True,True]
    
    :sigma,beta,phi_min,phi_max,lat_0,lon_0,n,gamma = array of integers
    :plot,dense_antipole,only_ocean = array of True/False
    
    :returns: list of lon,lat
    """
    
    # first check the number of grids given
    n_grids = np.size(sigma)
    #print("Number of grids: ", n_grids)

    if n_grids == 1:
        grid = svp_grid_one(sigma[0],beta[0],phi_min[0],phi_max[0],lat_0[0],lon_0[0],plot,dense_antipole,only_ocean)
        final_grid_lon = grid[0]
        final_grid_lat = grid[1]
    else:

        all_grids = []

        # Compute the different grids
        for i in range(0,n_grids):
            print('Grid {} of {}'.format(i+1,n_grids))
            grid_one = svp_grid_one(sigma[i],beta[i],phi_min[i],phi_max[i],lat_0[i],lon_0[i],
                                  plot,dense_antipole,only_ocean)
            all_grids.append(grid_one)
            # different grids are now stored in all_grids

        final_grid_lon = []
        final_grid_lat = []

        # New approach: put the initial grid into the above variables and while iterating over it, remove the
        # gridpoints are not needed. Once that loop is done, do another loop that goes through the first additional
        # grid and appends the points in that area to the variables. All this is inside a loop with length 0 to 
        # number of additional grids.


        final_grid_lon = all_grids[0][0]
        final_grid_lat = all_grids[0][1]
        


        for i in range(0,n_grids-1):
            # initialise distance array
            dist_deg_1 = []
            # first loop gets rid of points in the two variables above.
            for j in range(0,np.size(final_grid_lat)):

                dist_deg_1_var = spherical_distance_degrees(lat_0[i+1],lon_0[i+1],final_grid_lat[j],final_grid_lon[j])
                dist_deg_1.append(dist_deg_1_var)

            # get indices of points that are within area and remove points
            dist_deg_1_ind = np.where(np.array(dist_deg_1) <= gamma[i+1])
            final_grid_lon = np.delete(final_grid_lon,dist_deg_1_ind)
            final_grid_lat = np.delete(final_grid_lat,dist_deg_1_ind)
            
            # append points
            for k in range(0,np.size(all_grids[i+1][0])):

                dist_deg_2_var = spherical_distance_degrees(lat_0[i+1],lon_0[i+1],all_grids[i+1][1][k],all_grids[i+1][0][k])

                if dist_deg_2_var <= gamma[i+1]:
                    final_grid_lon = np.append(final_grid_lon,all_grids[i+1][0][k])
                    final_grid_lat = np.append(final_grid_lat,all_grids[i+1][1][k])
                else:
                    continue

        # plot
        if plot:
            plt.figure(figsize=(25,10))
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.coastlines()
            #ax.set_extent([-180,180,-90,90])

            plt.scatter(final_grid_lon,final_grid_lat,s=1,c='k',transform=ccrs.PlateCarree())
            plt.title('Final grid with {} gridpoints'.format(np.size(final_grid_lon)))
            plt.show(block=False)
            #plt.close()
        
    
    #print('Final number of gridpoints:',np.size(final_grid_lon))

    return list((final_grid_lat,final_grid_lon))
    


