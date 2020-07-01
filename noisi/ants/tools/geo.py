import os
from math import exp, pi, cos, sin, sqrt
from geographiclib import geodesic,geodesicline
import numpy as np


# Can approximate pieces of Earth surface area by spherical earth surface element 
# or by square lat-lon boxes on the ellipsoid. Quite similar results. 
def approx_surf_el(dlat,dlon,lat):
    # determine colatitude:
    colat = abs(lat-90.)
    # Radians
    colat = colat/180.*pi
    dlat = dlat/180.*pi
    dlon = dlon/180.*pi
    
    return(dlat*dlon*sin(colat))

def area_surfel(dlat,dlon,lat,r):
    surf_el = approx_surf_el(dlat,dlon,lat)
    
    return(r**2*surf_el)

def len_deg_lon(lat):
    
    (a,b,e_2) = wgs84()
    # This is the length of one degree of longitude 
    # approx. after WGS84, at latitude lat
    # in m
    lat = pi/180*lat
    dlon = (pi*a*cos(lat))/180*sqrt((1-e_2*sin(lat)**2))
    return round(dlon,2)

def len_deg_lat(lat):
    # This is the length of one degree of latitude 
    # approx. after WGS84, between lat-0.5deg and lat+0.5 deg
    # in m
    lat = pi/180*lat
    dlat = 111132.954 - 559.822 * cos(2*lat) + 1.175*cos(4*lat)
    return round(dlat,2)
    

def area_of_sqdeg(lat):
    # Give back the approx. area of a square degree at latitude lat
    # The sphericity of the Earth is not (yet) taken into account
    # This is a piecewise flat earth
    # in m^2
    l_lat = len_deg_lat(lat)
    l_lon = len_deg_lon(lat)
    if l_lat*l_lon == 0:
        area = 0.000001
    else:
        area = round(l_lat*l_lon,2)
    return area
        
    

def get_midpoint(lat1,lon1,lat2,lon2):
    
    """
    Obtain the coordinates of the point which is halfway between 
    point 1 (lat1, lon1) and point 2 (lat2, lon2) on great circle
    """
    
    p=geodesic.Geodesic.WGS84.Inverse(lat1,lon1,lat2,lon2)
    l=geodesic.Geodesic.WGS84.Line(p['lat1'],p['lon1'],p['azi1'])
    b=l.Position(0.5*p['s12'])

    return (b['lat2'],b['lon2'])
    

def get_antipode(lat,lon):
    
    if lon <= 0.:
        lon_a = 180. + lon
    else:
        lon_a = lon - 180.
    
    lat_a = -1. * lat
    
    return(lat_a,lon_a)
    
    
def wgs84():

    # semi-major axis, in m
    a = 6378137.0

    # semi-minor axis, in m
    b = 6356752.314245

    # inverse flattening f


    # squared eccentricity e
    e_2 = (a**2-b**2)/a**2
    
    return(a,b,e_2)