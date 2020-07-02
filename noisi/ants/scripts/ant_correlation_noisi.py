# Re-vamped correlation script
from __future__ import print_function
from mpi4py import MPI

import os
import time
#from noisi.ants.config import ConfigCorrelation
#cfg = ConfigCorrelation()

from noisi.ants.classes.corrblock_noisi import CorrBlock
from noisi.ants.tools.bookkeep import correlation_inventory
from obspy import UTCDateTime
from glob import glob
import pyasdf
from copy import deepcopy
# 'main':

# - import modules

# - determine own rank, size

# - initialize directories and report text files

#comm = MPI.COMM_WORLD
#rank = comm.Get_rank()
#size = comm.Get_size()
#print("Hello from rank %g" %rank)
#print("Size is %g" %size)





def correlate(cfg,comm,size,rank):    
  # Create output directory, if necessary

    #outdir = os.path.join('data','correlations')
    outdir = os.path.join(cfg.project_path,'data','correlations')
    
    if rank == 0 and not os.path.exists(outdir):
        os.mkdir(outdir)

    comm.Barrier()

    # Create own output directory, if necessary

    #rankdir = os.path.join(outdir,'rank_%g' %rank)
    #if not os.path.exists(rankdir):
    #    os.mkdir(rankdir)


    # correlation report file

    output_file = os.path.join(outdir,
        'correlation_report_rank%g.txt' %rank)

    if os.path.exists(output_file):
        ofid = open(output_file,'a')
        print('Resuming correlation job, Date:',file=ofid)
        print(time.strftime('%Y.%m.%dT%H:%M'),file=ofid)
    else:
        ofid = open(output_file,'w')
        print('Correlation job, Date:',file=ofid)
        print(time.strftime('%Y.%m.%dT%H:%M'),file=ofid)


    # - get list of files available;
    # - get blocks of channel pairs

    c = correlation_inventory(cfg)

# - LOOP over blocks:

    
    for b in c.blocks[rank::size]:

        block = deepcopy(b)
        # initialize a block of correlations
        c = CorrBlock(block,cfg)
        c.run(output_file = ofid)

# - append all the stationxmls to the asdf file, if asdf output is chosen

    if cfg.format_output.upper() == "ASDF" and rank == 0:
        filename = os.path.join('data','correlations','correlations.h5')

        with pyasdf.ASDFDataSet(filename) as ds:

            stations = []
            for cha in ds.auxiliary_data.CrossCorrelation.list():
                stations.append(cha.split('_')[0]+'.'+cha.split('_')[1])

            stations = set(stations)

            for sta in stations:
                ds.add_stationxml(os.path.join('meta','stationxml','%s.xml' %sta))

if __name__ == "__main__":
    correlate()


