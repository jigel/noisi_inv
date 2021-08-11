"""
This code is from the ants package: https://github.com/lermert/ants_2.

:copyright:
    noisi development team
:license:
    GNU Lesser General Public License, Version 3 and later
    (https://www.gnu.org/copyleft/lesser.html)

"""


import os
from glob import glob
from obspy import UTCDateTime
from copy import deepcopy


def find_files(indirs, format):

    content = list()

    for indir in indirs:

        content.extend(glob(os.path.join(indir, '*' + format.lower())))

        if format not in ['*', '???']:
            #print(os.path.join(indir, '*' + format.upper()))
            content.extend(glob(os.path.join(indir, '*' + format.upper())))
    content.sort()
    return content


def name_processed_file(stats, startonly=False):

    inf = [
        stats.network,
        stats.station,
        stats.location,
        stats.channel
    ]

    t1 = stats.starttime.strftime('%Y.%j.%H.%M.%S')
    t2 = stats.endtime.strftime('%Y.%j.%H.%M.%S')
    if startonly:
        t2 = '*'

    inf.append(t1)
    inf.append(t2)

    inf.append(stats._format)

    filenew = '{}.{}.{}.{}.{}.{}.{}'.format(*inf)

    return filenew


def name_correlation_file(sta1, sta2, corr_type, fmt='SAC'):

    name = '{}--{}.{}.{}'.format(sta1, sta2, corr_type, fmt)

    return(name)



def file_inventory(cfg):
    
    stations = {}
    data = {}
    readtimes = {}
    
    # start- and endtime specified in configuration
    t0 = UTCDateTime(cfg.time_begin) 
    t1 = UTCDateTime(cfg.time_end)

    # input directories and format (MSEED, SAC etc)
    indirs = cfg.indirs
    filefmt = cfg.input_format


    # list all files in input directories
    files = find_files(indirs,filefmt)
    
    for f in files:
        # decide whether file fits time range
        fn = os.path.basename(f).split('.')
        st = UTCDateTime('{}-{}T{}:{}:{}'.format(*fn[4:9]))
        et = UTCDateTime('{}-{}T{}:{}:{}'.format(*fn[9:14]))
       
        if st > t1 or et < t0:
            print("Data outside of chosen time range...")
            continue
        else:

            station = '{}.{}'.format(*fn[0:2])
            channel = '{}.{}.{}.{}'.format(*fn[0:4])

            # - stations dictionary: What stations exist and what channels do they have
            if station not in stations.keys():
                stations[station] = []

            # - channels dictionary: Inventory of files for each channel
            if channel not in stations[station]:
                stations[station].append(channel)

            if channel not in data.keys():
                data[channel] = []
                readtimes[channel] = []
                
            data[channel].append(f)
            readtimes[channel].append(st)

    return(stations, data, readtimes)


def station_pairs(staids,n,autocorr, only_autocorr=False):
   
    #staids = self.stations.keys()
    # sort alphabetically
    staids = list(staids) 
    staids.sort()
    blcks_stations = []
    #blcks_channels = []
    idprs = []

    n_ids = len(staids)
    n_auto = 0 if autocorr else 1
    #n_blk = cfg.n_stationpairs

    for i in range(n_ids):
        for j in range(i+n_auto,n_ids):

            if only_autocorr and i != j:
                continue

            if len(idprs) == n:
                blcks_stations.append(idprs)
                idprs = []

            idprs.append((staids[i],staids[j]))

    if len(idprs) <= n:
        blcks_stations.append(idprs)


    # idprs = []
# 
    # for blck in blcks_stations:
        # idprs = []
        # for pair in blck:
# 
            # idprs.extend(self._channel_pairs(pair[0],pair[1],cfg))  
# 
        # if idprs != []:
            # blcks_channels.append(idprs)

    return blcks_stations


def channel_pairs(channels1,channels2,cfg):


    channels = []
    tensor_comp = cfg.corr_tensorcomponents


    for c1 in channels1:
        for c2 in channels2:
            
            if cfg.update:
                f = name_correlation_file(c1,c2,cfg.corr_type)
                f = os.path.join('data','correlations',f)
                if os.path.exists(f):
                    continue

            loc1 = c1.split('.')[2]
            loc2 = c2.split('.')[2]

            if loc1 not in cfg.locations:
                continue

            if loc2 not in cfg.locations:
                continue

            if loc1 != loc2 and not cfg.locations_mix:
                continue

            comp = c1[-1] + c2[-1]
            
            if comp == 'EE' and cfg.rotate:
                comp = 'TT'

            if comp == 'NN' and cfg.rotate:
                comp = 'RR'

            if comp == 'EN' and cfg.rotate:
                comp = 'TR'

            if comp == 'NE' and cfg.rotate:
                comp = 'RT'

            if comp == 'NZ' and cfg.rotate:
                comp = 'RZ'

            if comp == 'ZN' and cfg.rotate:
                comp = 'ZR'

            if comp == 'EZ' and cfg.rotate:
                comp = 'TZ'

            if comp == 'ZE' and cfg.rotate:
                comp = 'ZT'

            if comp in tensor_comp:
                channels.append((c1,c2))

    return(channels)


class _block(object):

    def __init__(self):
        
        self.stations = []
        self.channels = []
        self.inventory = {}
        self.station_pairs = []
        self.channel_pairs = []

    def __repr__(self):

        return "Block containing %g channel pairs" %len(self.channel_pairs)

    def __str__(self):

        return "Block containing %g channel pairs" %len(self.channel_pairs)


class correlation_inventory(object):

    def __init__(self,cfg):

        self.cfg = cfg

        # - Find available data
        self.stations, self.files, self.readtimes = file_inventory(cfg)
        all_stations = self.stations.keys()

        # - Determine station pairs
        # - station pairs are grouped into blocks
        self.station_blocks = station_pairs(all_stations,
            cfg.n_stationpairs, cfg.corr_autocorr, cfg.corr_only_autocorr)


        self.blocks = []

        # - Determine channel pairs for each station pair
        # - station pairs are grouped into blocks
        for block in self.station_blocks:
            self._add_corrblock(block)

    def __repr__(self):
        return "Correlation inventory"

    def __str__(self):
        return "%g blocks in correlation inventory" %len(self.blocks)

    def _add_corrblock(self,station_block):

        block = _block()
        

        # Station pairs and channel pairs should be at the same index.
        block.station_pairs = station_block[:]
        block.channel_pairs = []

        # Make a unique station list for this block
        for p in station_block:
            block.stations.append(p[0])
            block.stations.append(p[1])
        block.stations = list(set(block.stations))

        # Find the relevant channel combinations
        for p in station_block:
            
            sta1 = p[0]
            sta2 = p[1]
            cpairs = channel_pairs(self.stations[sta1],
                    self.stations[sta2],self.cfg)
            block.channel_pairs.append(cpairs)

        # Make a unique channel list for this block
        for cp in block.channel_pairs:
            for c in cp:
                block.channels.append(c[0])
                block.channels.append(c[1])
        block.channels = list(set(block.channels))



        # Add file inventory for the channels in question
        inventory = {c: self.files[c] for c in block.channels}
        # add time stamps where files should be updated
        rtimes = []
        
        for c in block.channels:
            #print(c)
            rtimes.extend(self.readtimes[c])
        block.inventory = deepcopy(inventory)

        block.readtimes = rtimes
        block.readtimes.sort()
        #print(block.readtimes)
        
        # No channels are found if updating, and those pairs have already been computed.
        if block.channels != []: 
            self.blocks.append(block)
