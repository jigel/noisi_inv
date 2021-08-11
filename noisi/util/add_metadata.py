"""
Add metadata to cross-correlations

:copyright:
    noisi development team
:license:
    GNU Lesser General Public License, Version 3 and later
    (https://www.gnu.org/copyleft/lesser.html)
"""

import pandas as pd
from obspy import read
from obspy.geodetics import gps2dist_azimuth
import numpy as np
import sys
import os
from glob import glob

#indir = sys.argv[1]
#metafile = sys.argv[2]

import functools
print = functools.partial(print, flush=True)



def assign_geographic_metadata(indir, stationlistfile,comm,size,rank):

    traces = glob(os.path.join(indir, '*.SAC'))
    traces.extend(glob(os.path.join(indir, '*.sac')))
    
    if rank == 0:
        print(indir)
        print('Found traces:\n')
        print(traces[0])
        print('...to...')
        print(traces[-1])
        print('Assign geographical information.\n')

    meta = pd.read_csv(stationlistfile,keep_default_na=False,dtype=str)
    count_tr = 0
    
    # split up traces
    if rank == 0:
        traces_split = np.array_split(traces,size)
        traces_split = [k.tolist() for k in traces_split]
    else:
        traces_split = None
        
    traces_split = comm.scatter(traces_split,root=0) 
    
    for t in traces_split:
        if count_tr%100 == 0:
            print(f"At {count_tr} of {np.size(traces_split)} traces")

        tr = read(t)
        sta1 = str(os.path.basename(t).split('.')[1])
        try:
            sta2 = str(os.path.basename(t).split('--')[1].split('.')[1])
        except IndexError:
            sta2 = str(os.path.basename(t).split('.')[5])
        #print(sta1, sta2)
        lat1 = float(meta[meta['sta'] == sta1].iloc[0]['lat'])
        lat2 = float(meta[meta['sta'] == sta2].iloc[0]['lat'])
        lon1 = float(meta[meta['sta'] == sta1].iloc[0]['lon'])
        lon2 = float(meta[meta['sta'] == sta2].iloc[0]['lon'])
        #print(lat1, lon1, lat2, lon2)

        tr[0].stats.network = os.path.basename(t).split('.')[0]
        tr[0].stats.station = sta1
        tr[0].stats.location = ''
        tr[0].stats.channel = os.path.basename(t).split('.')[3].split('--')[0]
        tr[0].stats.sac.stlo = lon1
        tr[0].stats.sac.stla = lat1
        tr[0].stats.sac.evlo = lon2
        tr[0].stats.sac.evla = lat2
        tr[0].stats.sac.kuser0 = meta[meta['sta'] == sta2].iloc[0]['net']

        tr[0].stats.sac.kevnm = sta2
        tr[0].stats.sac.kuser1 = ''
        try:
            tr[0].stats.sac.kuser2 = os.path.basename(t).split('--')[1].\
                split('.')[3]
        except IndexError:
            sta2 = os.path.basename(t).split('.')[7]
        tr[0].stats.sac.user0 = 1
        geoinf = gps2dist_azimuth(lat1, lon1, lat2, lon2)
        tr[0].stats.sac.dist = geoinf[0]
        tr[0].stats.sac.az = geoinf[1]
        tr[0].stats.sac.baz = geoinf[2]

        tr.write(t, format='SAC')

        count_tr += 1
        
    comm.barrier()

#if __name__ == '__main__':
#    assign_geographic_metadata(indir, metafile)
