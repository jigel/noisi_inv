"""
Geographical functions to compute distances, surface areas and more

:copyright:
    noisi development team
:license:
    GNU Lesser General Public License, Version 3 and later
    (https://www.gnu.org/copyleft/lesser.html)
"""

import numpy as np
from math import pi, sin, cos, sqrt
import pandas as pd
import os 
from obspy.geodetics import gps2dist_azimuth
try:
    import cartopy.io.shapereader as shpreader
    import shapely.geometry as sgeom
    from shapely.ops import unary_union
    from shapely.prepared import prep
    from shapely.geometry import Polygon
except ImportError:
    pass
from warnings import warn


def geographical_distances(grid, location):

    def f(lat, lon, location):

        return abs(gps2dist_azimuth(lat, lon, location[0], location[1])[0])

    dist = np.array([f(lat, lon, location) for lat, lon in
                     zip(grid[1], grid[0])])
    return dist


def is_land(x, y, res="110m"):

    if 'prep' not in globals():
        raise ImportError("cartopy is needed to design ocean-only source.")
    assert(res in ["10m", "50m", "110m"]), "Resolution must be 10m, 50m, 110 m"

    land_shp_fname = shpreader.natural_earth(resolution=res,
                                             category='physical',
                                             name='land')

    land_geom = unary_union(list(shpreader.Reader(land_shp_fname).
                                 geometries()))
     
    # add Caspian Sea and Black Sea
    #outline_casp = np.asarray([[27,40],[60,30],[60,50],[23,50],[27,40]])
    outline_casp = np.asarray([[35,5],[65,25],[60,50],[23,50],[35,5]])

    casp_poly = Polygon((outline_casp))
    casp = prep(casp_poly)
    
    outline_hud = np.asarray([[-80,48],[-105,58],[-77,74],[-58,53],[-80,48]])
    hud_poly = Polygon((outline_hud))
    hud = prep(hud_poly)
    
    
    land = prep(land_geom)
    is_land = np.zeros(len(x))
    
    for i in range(len(x)):
        is_land[i] = land.contains(sgeom.Point(x[i], y[i]))
        #if land.contains(sgeom.Point(x[i], y[i])) or casp.contains(sgeom.Point(x[i], y[i])) or hud.contains(sgeom.Point(x[i], y[i])):
        #    is_land[i] = 1
        
    return is_land


def wgs84():

    # semi-major axis, in m
    a = 6378137.0

    # semi-minor axis, in m
    b = 6356752.314245

    # inverse flattening f
    f = a / (a - b)

    # squared eccentricity e
    e_2 = (a ** 2 - b ** 2) / a ** 2

    return(a, b, e_2, f)


def geograph_to_geocent(theta):
    # geographic to geocentric
    # https://en.wikipedia.org/wiki/Latitude#Geocentric_latitude
    e2 = wgs84()[2]
    theta = np.rad2deg(np.arctan((1 - e2) * np.tan(np.deg2rad(theta))))
    return theta


def geocent_to_geograph(theta):
    # the other way around
    e2 = wgs84()[2]
    theta = np.rad2deg(np.arctan(np.tan(np.deg2rad(theta)) / (1 - e2)))
    return(theta)


def len_deg_lon(lat):
    (a, b, e_2, f) = wgs84()

    # This is the length of one degree of longitude
    # approx. after WGS84, at latitude lat
    # in m
    lat = pi / 180 * lat
    dlon = (pi * a * cos(lat)) / 180 * sqrt((1 - e_2 * sin(lat) ** 2))
    return round(dlon, 5)


def len_deg_lat(lat):
    # This is the length of one degree of latitude
    # approx. after WGS84, between lat-0.5deg and lat+0.5 deg
    # in m
    lat = pi / 180 * lat
    dlat = 111132.954 - 559.822 * cos(2 * lat) + 1.175 * cos(4 * lat)
    return round(dlat, 5)


def get_spherical_surface_elements(lon, lat, r=6.378100e6):

    if len(lon) < 3:
        raise ValueError('Grid must have at least 3 points.')
    if len(lon) != len(lat):
        raise ValueError('Grid x and y must have same length.')

    # surfel
    surfel = np.zeros(lon.shape)
    colat = 90. - lat

    # find neighbours
    for i in range(len(lon)):

        # finding the relevant neighbours is very specific to how
        # the grid is set up here (in rings of constant colatitude)!
        # get the nearest longitude along the current colatitude
        current_colat = colat[i]
        if current_colat in [0., 180.]:
            # surface area will be 0 at poles.
            continue

        colat_idx = np.where(colat == current_colat)
        lon_idx_1 = np.argsort(np.abs(lon[colat_idx] - lon[i]))[1]
        lon_idx_2 = np.argsort(np.abs(lon[colat_idx] - lon[i]))[2]
        closest_lon_1 = lon[colat_idx][lon_idx_1]
        closest_lon_2 = lon[colat_idx][lon_idx_2]

        if closest_lon_1 > lon[i] and closest_lon_2 > lon[i]:
            d_lon = np.abs(min(closest_lon_2, closest_lon_1) - lon[i])

        elif closest_lon_1 < lon[i] and closest_lon_2 < lon[i]:
            d_lon = np.abs(max(closest_lon_2, closest_lon_1) - lon[i])

        else:
            if closest_lon_1 != lon[i] and closest_lon_2 != lon[i]:
                d_lon = np.abs(closest_lon_2 - closest_lon_1) * 0.5
            else:
                d_lon = np.max(np.abs(closest_lon_2 - lon[i]),
                               np.abs(closest_lon_1 - lon[i]))

        colats = np.array(list(set(colat.copy())))
        colat_idx_1 = np.argsort(np.abs(colats - current_colat))[1]
        closest_colat_1 = colats[colat_idx_1]
        colat_idx_2 = np.argsort(np.abs(colats - current_colat))[2]
        closest_colat_2 = colats[colat_idx_2]

        if (closest_colat_2 > current_colat
            and closest_colat_1 > current_colat):

            d_colat = np.abs(min(closest_colat_1,
                                 closest_colat_2) - current_colat)

        elif (closest_colat_2 < current_colat and
              closest_colat_1 < current_colat):
            d_colat = np.abs(max(closest_colat_1,
                                 closest_colat_2) - current_colat)

        else:
            if (closest_colat_2 != current_colat
                and closest_colat_1 != current_colat):
                d_colat = 0.5 * np.abs(closest_colat_2 - closest_colat_1)
            else:
                d_colat = np.max(np.abs(closest_colat_2 - current_colat),
                                 np.abs(closest_colat_1 - current_colat))

        surfel[i] = (np.deg2rad(d_lon) *
                     np.deg2rad(d_colat) *
                     sin(np.deg2rad(colat[i])) * r ** 2)

    return(surfel)


def points_on_ell(dx, xmin=-180., xmax=180., ymin=-89.999, ymax=89.999):
    """
    Calculate an approximately equally spaced grid on an
    elliptical Earth's surface.
    :type dx: float
    :param dx: spacing in latitudinal and longitudinal direction in meter
    :returns: np.array(latitude, longitude) of grid points,
    where -180<=lon<180     and -90 <= lat < 90
    """

    if xmax <= xmin or ymax <= ymin:
        msg = 'Lower bounds must be lower than upper bounds.'
        raise ValueError(msg)
    assert xmax <= 180., 'Longitude must be within -180 -- 180 degrees.'
    assert xmin >= -180., 'Longitude must be within -180 -- 180 degrees.'

    gridx = []
    gridy = []

    if ymin == -90.:
        ymin = -89.999
        warn("Resetting lat_min to -89.999 degree")
    if ymax == 90.:
        ymax = 89.999
        warn("Resetting lat_max to 89.999 degree")
    lat = ymin
    # do not start or end at pole because 1 deg of longitude is 0 m there
    while lat <= ymax:
        d_lon = dx / len_deg_lon(lat)
        # the starting point of each longitudinal circle is randomized
        perturb = np.random.rand(1)[0] * d_lon - 0.5 * d_lon
        lon = min(max(xmin + perturb, -180.), 180.)

        while lon <= xmax:
            gridx.append(lon)
            gridy.append(lat)
            lon += d_lon
        d_lat = dx / len_deg_lat(lat)
        lat += d_lat
    return list((gridx, gridy))



def get_voronoi_surface_area(grd,config,sigma,beta,phi_min,phi_max,lat_0,lon_0,gamma,output="project"):
    """
    Computes the voronoi cell surface area of a grid on a sphere.
    If the grid is set to only be in the ocean it will be recomputed with all gridpoints. 
    The gridpoints and corresponding voronoi cells are removed thereafter to avoid unrealistic cells along the coastline.
    
    :type grd: array of lat,lon
    :type only_ocean: bool
    
    """
    
    print("Computing Voronoi cell surface areas..")
    
    # import borrowed functions
    from noisi.borrowed_functions.voronoi_polygons import getVoronoiCollection
    from noisi.borrowed_functions.voronoi_surface_area import calculate_surface_area_of_a_spherical_Voronoi_polygon
    from noisi.borrowed_functions.voronoi_polygons import xyzToSpherical
        
    if config["svp_only_ocean"]:
        
        # compute full grid
        from noisi.util.source_grid_svp import svp_grid
        
        grd = svp_grid(sigma=sigma,
                          beta=beta,
                          phi_min=phi_min,
                          phi_max=phi_max,
                          lat_0=lat_0,
                          lon_0=lon_0,
                          gamma=gamma,
                          plot=config['svp_plot'],
                          dense_antipole=config['svp_dense_antipole'],
                          only_ocean=False)
        
        
    # convert grid into panda dataframe
    gridpd = {'lat': grd[1], 'lon': grd[0]}
    grid_data = pd.DataFrame(data=gridpd)
    
    # Calculate the vertices for the voronoi cells
    voronoi = getVoronoiCollection(data=grid_data,lat_name='lat',lon_name='lon',full_sphere=True)
    
    # Calculate the surface area for each voronoi cell
    voronoi_lat = []
    voronoi_lon = []
    voronoi_area = []
    
    for i in range(0,np.size(voronoi.points,0)):
        P_cart = xyzToSpherical(x=voronoi.points[i,0],y=voronoi.points[i,1],z=voronoi.points[i,2])
        voronoi_lat.append(P_cart[0])
        voronoi_lon.append(P_cart[1])
        vert_points = voronoi.vertices[voronoi.regions[i]]
        area = calculate_surface_area_of_a_spherical_Voronoi_polygon(vert_points,6371)
        voronoi_area.append(area)
        #if i%1000 == 0:
        #    print('%g of %g voronoi cell surface areas calculated.' %(i,np.size(voronoi.points,0)),flush=True)
        
    # Reassign grd so that everything is in the right order
    grd = np.asarray([voronoi_lon,voronoi_lat])
    voronoi_area = np.asarray(voronoi_area)
    print('All voronoi cell surface areas calculated.')
    
    
    if config["svp_only_ocean"]:
        print("Removing voronoi cells on land..")
        
        
        grid_onlyocean_lon = []
        grid_onlyocean_lat = []
        voronoi_area_onlyocean = []
        
        is_ocean = np.abs(is_land(grd[0],grd[1]) - 1.)
        
        for i in range(np.size(grd[0])):
            if is_ocean[i] == 1:
                grid_onlyocean_lon.append(grd[0][i])
                grid_onlyocean_lat.append(grd[1][i])
                voronoi_area_onlyocean.append(voronoi_area[i])
            else:
                continue
                
        print('Gridpoints and voronoi cells on land removed.')
            
        grd = np.asarray([grid_onlyocean_lon,grid_onlyocean_lat])
        surf_area = np.asarray(voronoi_area_onlyocean)        
        
    else:
        surf_area = voronoi_area
        
        
    return grd,surf_area




