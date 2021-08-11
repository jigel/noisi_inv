"""
Download pre-computed wavefields from syngine

:copyright:
    noisi development team
:license:
    GNU Lesser General Public License, Version 3 and later
    (https://www.gnu.org/copyleft/lesser.html)
"""


# script to download pre-computed wavefields from Syngine: http://ds.iris.edu/ds/products/syngine/
# best wavefields for local computation:
# prem_a_20s = 0.9GB
# prem_a_10s = 86GB
# prem_a_5s = 123GB


import numpy as np
import os
import requests
import sys
import time


def syngine_download(wf_name='prem_a_20s',output_path='./',check = True):
    """
    script to download pre-computed wavefields from Syngine: http://ds.iris.edu/ds/products/syngine/
    
    best wavefields for local computation:
    prem_a_20s = 0.9GB (default)
    prem_a_10s = 86GB
    prem_a_5s = 123GB
    
    output_path :: output folder, defaults to script folder
    
    check :: checks wavefield size and asks for user input for downloading. Set to False to automatically download
    """

    # link to syngine wavefield
    syngine_link = f'http://ds.iris.edu/files/syngine/axisem/models/{wf_name}_merge_compress2/merged_output.nc4'

    # folder for wavefield
    wf_path = os.path.abspath(os.path.join(output_path,wf_name))

    # check file size
    response = requests.get(syngine_link, stream=True)
    total = response.headers.get('content-length')
    db_size = float(total)*10**-9

    print(f"Database size: {np.around(db_size,3)} GB")

    while True:
        
        # if database too small it wasn't found properly
        if db_size < 0.001:
            print('No database found. Check link.')
            break

        # if check: user input if it should continue
        if check:
            print(f"Do you want to download the {wf_name} database with {np.around(db_size,3)} GB? yes / no")
            continue_dw = input()
        else:
            continue_dw = 'yes'

        if continue_dw in ['yes','Yes','y']:
            print(f'Database will be downloaded and saved in {wf_path}/')

            if not os.path.exists(wf_path):
                os.makedirs(wf_path)

            database_file = os.path.join(wf_path,'merged_output.nc4')

            # download file in chunks using iter_content and output progress bar
            with open(database_file, 'wb') as f:
                start = time.time()
                response = requests.get(syngine_link, stream=True)
                total = response.headers.get('content-length')

                if total is None:
                    f.write(response.content)
                else:
                    downloaded = 0
                    total = int(total)
                    for data in response.iter_content(chunk_size=max(int(total/1000), 1024*1024)):
                        downloaded += len(data)
                        f.write(data)
                        done = int(50*downloaded/total)
                        sys.stdout.write(f"\r {np.around((downloaded/total)*100,1)} % [{'█' * done}{'-' * (50-done)}] {np.around(downloaded*1e-6,1)}/{np.around(total*1e-6,1)}MB ")
                        sys.stdout.flush()
            sys.stdout.write('\n')     
            
            if downloaded == total:
                dw_success = True
                print(f"Syngine wavefield successfully downloaded and saved in {wf_path}.")
            else:
                dw_success = False
                print(f"Problem downloading syngine wavefield. Check wavefield file.")

            break
            
        elif continue_dw in ['no','No','n']:
            dw_success = False
            print('Database will not be downloaded.')
            break

        else:
            continue
            
            
    return dw_success, wf_path


if __name__ == '__main__':
    syngine_download(wf_name='prem_a_20s',output_path='./',check = True)