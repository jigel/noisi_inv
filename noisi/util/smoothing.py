"""
Smoothing routine for noisi
:copyright:
    noisi development team
:license:
    GNU Lesser General Public License, Version 3 and later
    (https://www.gnu.org/copyleft/lesser.html)
"""
import numpy as np
from math import sqrt, pi
import sys

def get_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1 = np.deg2rad(lon1)
    lat1 = np.deg2rad(lat1)
    lon2 = np.deg2rad(lon2)
    lat2 = np.deg2rad(lat2)

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c
    return km*1000

def smooth_gaussian(values, coords, rank, size, sigma, r=6371000.,
                    threshold=1e-16):
    # coords format: (lon,lat)

    v_smooth = np.zeros(values.shape)

    a = 1. / (sigma * sqrt(2. * pi))
    for i in range(rank, len(values), size):

        lat1 = coords[1][i]
        lon1 = coords[0][i]
        lat2 = coords[1]
        lon2 = coords[0]
        
        dist = get_distance(lat1,lon1,lat2,lon2)
        
        weight = a * np.exp(-(dist) ** 2 / (2 * sigma ** 2))
        idx = weight >= threshold
        v_smooth[i] = np.sum(np.multiply(weight[idx], values[idx])) / idx.sum()
    
    return v_smooth


def apply_smoothing_sphere(rank, size, values, coords, sigma, cap,
                           threshold, comm):

    sigma = float(sigma)
    cap = float(cap)
    threshold = float(threshold)

    # clip
    perc_up = np.percentile(values, cap, overwrite_input=False)
    perc_dw = np.percentile(values, 100 - cap, overwrite_input=False)
    values = np.clip(values, perc_dw, perc_up)

    # get the smoothed map; could use other functions than Gaussian here
    v_s = smooth_gaussian(values, coords, rank, size, sigma,
                          threshold=threshold)

    comm.barrier()

    # collect the values
    v_s_all = comm.gather(v_s, root=0)
    # rank 0: save the values
    if rank == 0:
        #print('Gathered.')
        v_s = np.zeros(v_s.shape)
        for i in range(size):
            v_s += v_s_all[i]

        return(v_s)


def smooth(inputfile, outputfile, coordfile, sigma, cap, thresh, comm, size,
           rank):

    for ixs in range(len(sigma)):
        sigma[ixs] = float(sigma[ixs])

    coords = np.load(coordfile)
    values = np.array(np.load(inputfile), ndmin=2)
    if values.shape[0] > values.shape[-1]:
        values = np.transpose(values)
    smoothed_values = np.zeros(values.shape)
    for i in range(values.shape[0]):
        array_in = values[i, :]
        try:
            sig = sigma[i]
        except IndexError:
            sig = sigma[-1]

        v = apply_smoothing_sphere(rank, size, array_in,
                                   coords, sig, cap, threshold=thresh,
                                   comm=comm)
        comm.barrier()

        if rank == 0:
            smoothed_values[i, :] = v

    comm.barrier()

    if outputfile is not None:
        if rank == 0:
            np.save(outputfile, smoothed_values)
            return()
        else:
            return()
    else:
            return(smoothed_values)


if __name__ == '__main__':

    # pass in: input_file, output_file, coord_file, sigma
    # open the files
    inputfile = sys.argv[1]
    outputfile = sys.argv[2]
    coordfile = sys.argv[3]
    sigma = sys.argv[4].split(',')
    cap = float(sys.argv[5])

    try:
        thresh = float(sys.argv[6])
    except IndexError:
        thresh = 1.e-12

    smooth(inputfile, outputfile, coordfile, sigma, cap, thresh)
