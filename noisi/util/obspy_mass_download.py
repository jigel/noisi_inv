import obspy
import numpy as np
from pandas import read_csv
from obspy import UTCDateTime
import os
from glob import glob
import csv

import functools
print = functools.partial(print, flush=True)


from obspy.clients.fdsn.mass_downloader import RectangularDomain, Restrictions, MassDownloader,CircularDomain, GlobalDomain

# not parallel  but runs multiple threads
def obspy_mass_downloader(args):
    """
    Uses the obspy mass_downloader to download available data.
    Can either take a stationlist or domain.
    """
    
    
    # make new folder in project folder where raw data and inventory is saved
    data_folder = os.path.join(args.project_path,'data')
    
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        print(f"Created folder for data in project: {data_folder}")
    
    # make data folder
    data_path = os.path.join(data_folder,'raw')

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # make inv folder
    inv_path = os.path.join(data_folder,'inv')

    if not os.path.exists(inv_path):
        os.makedirs(inv_path)

    # choose date for which data should be downloaded
    t_start = args.t_start
    t_end = args.t_end
        
        
    print(f"Start time: {t_start}")
    print(f"End time: {t_end}")
    
    
    # read stationlist and get net sta 
    if args.stationlist is not None:
        # load stationlist
        stationlist = read_csv(args.stationlist,keep_default_na=False)
        net = ','.join(set(stationlist['net']))
        sta = ','.join(set(stationlist['sta']))
        # set to global so that everything is checked
        domain_type = 'global'

    else:
        stationlist = None
        net = '*'
        sta = '*'
        
        domain_type = args.domain_type # 'rectangular','global'

        
    # check if domain_type is one of circular, rectangular, global  
    if domain_type.lower() not in ['circular','rectangular','global']:
        e = "domain_type has to be 'circular', 'rectangular, or 'global'"
        raise Exception(e)
        
    
    # minimum station distance
    min_stat_dist_deg = args.min_station_dist
    min_stat_dist_m = min_stat_dist_deg*111000

    # parameters for rectangular
    if domain_type.lower() == 'rectangular':
        lat_min = args.rect_lat_min
        lat_max = args.rect_lat_max
        lon_min = args.rect_lon_min
        lon_max = args.rect_lon_max

        domain = RectangularDomain(minlatitude=lat_min,
                                   maxlatitude=lat_max,
                                   minlongitude=lon_min,
                                   maxlongitude=lon_max)


    if domain_type.lower() == 'circular':
        # useful tool: https://www.mapdevelopers.com/draw-circle-tool.php
        lat = args.circ_lat_center
        lon = args.circ_lon_center
        r_min = args.circ_radius_min
        r_max = args.circ_radius_max

        domain = CircularDomain(latitude=lat,
                                longitude=lon,
                                minradius=r_min,
                                maxradius=r_max)

    if domain_type.lower() == 'global':

        domain = GlobalDomain()

    
    cha = ','.join(args.download_data_channels)

    # get restrictions for downloader
    restrictions = Restrictions(
        starttime = t_start,
        endtime = t_end,
        # daily chunks
        chunklength_in_sec=86400,
        network=net,station=sta,location='*',channel=cha,
        reject_channels_with_gaps=True,
        minimum_length=0.0,
        minimum_interstation_distance_in_m=min_stat_dist_m,
        sanitize=True)


    # get mass downloader and use all providers (in better order)
    mdl = MassDownloader(providers=["IRIS","ORFEUS","ETH","BGR","USGS","GFZ",
                          "EMSC","GEONET","ICGC","INGV","IPGP","KOERI",
                          "LMU","NCEDC","NIEP","NOA","RESIF","SCEDC","TEXNET","USP"])

    mdl.download(domain,restrictions,mseed_storage=data_path,stationxml_storage=inv_path,print_report=False)

    
    # check how many files
    print(f"Found {np.size(os.listdir(inv_path))} stations with data.")
    
    # remove all but one location for each station
    station_found = []

    for file in os.listdir(data_path):
        net_1 = file.split('.')[0]
        sta_1 = file.split('.')[1]
        cha_1 = file.split('.')[3]

        sta_name = f"{net_1}.{sta_1}..{cha_1}"

        if sta_name not in station_found:
            station_found.append(sta_name)
        else:
            os.remove(os.path.join(data_path,file))
    
    if np.size(os.listdir(inv_path)) == []:
        raise Exception("Could not find any data")
   
    if args.stationlist is None: 
        # turn it into stationlist
        stations_csv_final = [['net','sta','lat','lon']]

        # loop over xml file and make stationlist file
        station_xml_files = glob(os.path.join(inv_path,"*xml"))
        stationlist_output = os.path.join(args.project_path,'stationlist_data.csv')


        for station in station_xml_files:
            inv_var = obspy.read_inventory(station)

            net = os.path.basename(station).split(".")[0]
            sta = os.path.basename(station).split(".")[1]
            lat = inv_var.get_coordinates(inv_var.get_contents()['channels'][0])['latitude']
            lon = inv_var.get_coordinates(inv_var.get_contents()['channels'][0])['longitude']

            stations_csv_final.append([net,sta,lat,lon])


        print(f"Found data and inventory for {np.size(stations_csv_final,0)-1} stations.")
        print(f"Raw data can be found in: {data_path}")
        print(f"Inventory can be found in: {inv_path}")

        with open(stationlist_output, 'w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerows(stations_csv_final)
        csvFile.close()


        print(f"Data stationlist saved as {stationlist_output}")

    else:
        stationlist_output = args.stationlist 
        print(f"Raw data can be found in: {data_path}")
        print(f"Inventory can be found in: {inv_path}")     

    return stationlist_output
