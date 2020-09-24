# convert sac to asdf and asdf to obspy Stream
import obspy
import pyasdf
import os
import h5py
import numpy as np

def sac_to_asdf(files,asdf_filepath, n=4):
    '''
    Input: list of .sac/.SAC files, asdf_file_path, n (number of characters to remove from end of sac file to get station pair)
    Output: asdf_file 
    Returns: path to asdf_file
    '''

    asdf_file = pyasdf.ASDFDataSet(asdf_filepath, mode='a',mpi=False)

    asdfmeta = {}

    for file in files:

        tr = obspy.read(file)[0]

        stat_pair = os.path.basename(file)[:-n]

        stat_1 = stat_pair.split("--")[0].replace(".", "_")
        stat_2 = stat_pair.split("--")[1].replace(".", "_")

        # get metadata from stats
        for i in tr.stats:
            if i == "sac":
                for j in tr.stats['sac']:
                    asdfmeta[f"stats.sac.{j}"] = tr.stats['sac'][j]
            else:
                # h5 can't write obspy Datetime object
                if i in ['starttime','endtime']:
                    asdfmeta[f"stats.{i}"] = str(tr.stats[i])
                else:
                    asdfmeta[f"stats.{i}"] = tr.stats[i]

        asdf_file.add_auxiliary_data(tr.data,data_type="CrossCorrelation",path=f"{stat_1}/{stat_2}",parameters=asdfmeta)
        
    return asdf_filepath


def asdf_to_stream(asdf_filepath):
    '''
    Input: path to asdf file
    Return: obspy Stream
    '''
        
    asdf_file = pyasdf.ASDFDataSet(asdf_filepath,mode='r',mpi=False)
    st = obspy.Stream()

    for i in asdf_file.auxiliary_data.CrossCorrelation.list():
        for j in asdf_file.auxiliary_data.CrossCorrelation[i].list():

            data_var = asdf_file.auxiliary_data.CrossCorrelation[i][j].data[()]

            parameters = asdf_file.auxiliary_data.CrossCorrelation[i][j].parameters

            header_var = {}
            header_var['sac'] = obspy.core.util.attribdict.AttribDict()

            for k in parameters:
                val = parameters[k]
                if k.split('.')[1] == 'sac':
                    header_var['sac'][k.split('.')[2]] = val
                else:
                    if k.split('.')[1] in ['starttime','endtime']:
                        header_var[k.split('.')[1]] = obspy.UTCDateTime(val)
                    else:
                        header_var[k.split('.')[1]] = val

            tr_var = obspy.Trace(data=data_var,header=header_var)
            
            st += tr_var
    
    return st


def npy_to_h5(files,h5_filepath,n=4):
    '''
    Input: list of .npy filepaths, output h5_filepath, n (number of characters to remove)
    Output: h5_file
    Returns: h5_filepath
    '''
    
    file_h5 = h5py.File(h5_filepath,'a')
    
    for file in files:
        filename = os.path.basename(file)[:-n]
        
        file_h5.create_dataset(filename,data=np.load(file))
        
    file_h5.close()
    
    return h5_filepath

