"""
Check possible correlation pairs

:copyright:
    noisi development team
:license:
    GNU Lesser General Public License, Version 3 and later
    (https://www.gnu.org/copyleft/lesser.html)
"""


from glob import glob
import os
from pandas import read_csv
import yaml


def define_correlationpairs(proj_dir, auto_corr=False,
                            only_observed=True, channel='*'):
    """
    Match correlation pairs.
    :param proj_dir: Path to project directory,
    where a stations.txt file has to be located
    :param auto_corr: Include or exclude autocorrelations
    :return corr_pairs: A list of correlation pairs in the format
    [['net1 sta1 lat1 lon1','net2 sta2 lat2 lon2'],[...]]
    """

    try:
        stations = read_csv(os.path.join(proj_dir, 'stationlist.csv'),keep_default_na=False)
        nets = list(stations.net.values)
        stations = list(stations.sta.values)
        stations = [str(nets[i]) + '  ' + str(stations[i])
                    for i in range(len(stations))]

    except IOError:
        stations = open(os.path.join(proj_dir, 'stations.txt'), 'r')
        stations = stations.read().split('\n')

    stations.sort()
    i = 0
    corr_pairs = []
    
    conf = yaml.safe_load(open(os.path.join(proj_dir,
                                            'config.yml')))
    
    
    if conf['wavefield_channel'] == 'all':
        channel = ['??Z','??E','??N']
        channel_list = False
    # Added if wavefield_channel is a list of pairs
    elif isinstance(conf['wavefield_channel'],list):
        channels_flipped = [i[::-1] for i in conf['wavefield_channel'] if i[0] != i[1]] # making sure both directions are in there
        conf['wavefield_channel'] = list(set(conf['wavefield_channel'] + channels_flipped))
        channel = ['??Z','??E','??N']
        channel_list = True
    else:
        channel = conf['wavefield_channel']
        channel = ['??' + channel[-1]]
        channel_list = False

    #print(channel)
    
    while i < len(stations):
        for cha1 in channel:
            for cha2 in channel:
                sta_0 = stations[i].strip()
                if auto_corr:
                    stas_other = stations[i:]
                else:
                    stas_other = stations[i + 1:]
                #i += 1

                for sta in stas_other:

                    if '' in [sta_0, sta]:
                        continue
                        
                    if channel_list:
                        if not f"{cha1.split('?')[-1]}{cha2.split('?')[-1]}" in conf['wavefield_channel']:
                            continue
                        if not f"{cha2.split('?')[-1]}{cha1.split('?')[-1]}" in conf['wavefield_channel']:
                            continue 
                            
                    corr_pairs.append([str(sta_0)+' '+str(cha1), str(sta)+ ' '+str(cha2)])
                    
        i += 1
                    
    corrpairs_unique = sorted([list(x) for x in set(tuple(x) for x in corr_pairs)])
    corr_pairs = corrpairs_unique

    #print('corr',corr_pairs)
    
    
    return corr_pairs


def rem_no_obs(stapairs, source_conf, directory, ignore_network=True):

    conf = yaml.safe_load(open(os.path.join(source_conf['project_path'],
                                            'config.yml')))
    
    #if conf['wavefield_channel'] == 'all':
    #    channel = ['??Z','??E','??N']
    #else:
    #    channel = conf['wavefield_channel']
    #    channel = ['??' + channel[-1]]
        
    stapairs_new = []
    for i in range(len(stapairs)):
        # Check if an observation is actually available
        if stapairs[i] == '':
            break

        sta1 = '{}.{}.*.{}'.format(*(stapairs[i][0].split()[0: 2] + [stapairs[i][0].split()[-1]]))
        sta2 = '{}.{}.*.{}'.format(*(stapairs[i][1].split()[0: 2] + [stapairs[i][1].split()[-1]]))
        p_new = glob_obs_corr(sta1, sta2, directory, ignore_network)

        if p_new == []:
            continue
        stapairs_new.append(stapairs[i])


    # unique pairs
    stapairs_unique = [list(x) for x in set(tuple(x) for x in stapairs_new)]
    stapairs_new = stapairs_unique
                
    return stapairs_new


def rem_fin_prs(stapairs, source_conf, step):

    """
    Remove those station pairs from the list for which correlation / kernel
    has already been calculated.
    :param sta_pairs: List of all station pairs
    :param source_conf: source config dictionary
    :param step: step nr
    """

    conf = yaml.safe_load(open(os.path.join(source_conf['project_path'],
                                            'config.yml')))
    
    #if conf['wavefield_channel'] == 'all':
    #    channel = ['MXZ','MXE','MXN']
    #else:
    #    channel = ['MX' + conf['wavefield_channel'][-1]]
        
    #channel = 'MX' + conf['wavefield_channel']

    mod_dir = os.path.join(source_conf['source_path'],
                           'iteration_{}'.format(step), 'corr')

    stapairs_new = []

    for sp in stapairs:
        
        id1 = sp[0].split()[0] + sp[0].split()[1]
        id2 = sp[1].split()[0] + sp[1].split()[1]

        if id1 < id2:
            inf1 = sp[0].split()
            inf2 = sp[1].split() 
        else:
            inf2 = sp[0].split()
            inf1 = sp[1].split()

        sta1 = "{}.{}..MX{}".format(*(inf1[0: 2] + [sp[0].split()[2][-1]]))
        sta2 = "{}.{}..MX{}".format(*(inf2[0: 2] + [sp[1].split()[2][-1]]))

        
        corr_name = "{}--{}.sac".format(sta1, sta2)
        corr_name = os.path.join(mod_dir, corr_name)
        
        if not os.path.exists(corr_name):
            stapairs_new.append([sp[0][:-4]+' MX'+sp[0].split()[2][-1],sp[1][:-4]+' MX'+sp[1].split()[2][-1]])


    # unique pairs
    stapairs_unique = [list(x) for x in set(tuple(x) for x in stapairs_new)]
    stapairs_new = stapairs_unique
                    
    return stapairs_new


def get_synthetics_filename(obs_filename, dir, synth_location='',
                            fileformat='sac', synth_channel_basename='??',
                            ignore_network=True, verbose=False):

    inf = obs_filename.split('--')

    if len(inf) == 1:
        # old station name format
        inf = obs_filename.split('.')
        net1 = inf[0]
        sta1 = inf[1]
        cha1 = inf[3]
        net2 = inf[4]
        sta2 = inf[5]
        cha2 = inf[7]
    elif len(inf) == 2:
        # new station name format
        inf1 = inf[0].split('.')
        inf2 = inf[1].split('.')
        net1 = inf1[0]
        sta1 = inf1[1]
        net2 = inf2[0]
        sta2 = inf2[1]
        cha1 = inf1[3]
        cha2 = inf2[3]

    cha1 = synth_channel_basename + cha1[-1]
    cha2 = synth_channel_basename + cha2[-1]

    sfilename = None
    if ignore_network:
        synth_filename1 = '*.{}.{}.{}--*.{}.{}.{}.{}'.format(sta1,
                                                             synth_location,
                                                             cha1, sta2,
                                                             synth_location,
                                                             cha2, fileformat)
        synth_filename2 = '*.{}.{}.{}--*.{}.{}.{}.{}'.format(sta2,
                                                             synth_location,
                                                             cha2, sta1,
                                                             synth_location,
                                                             cha1, fileformat)

        try:
            sfilename = glob(os.path.join(dir, synth_filename1))[0]
        except IndexError:
            try:
                sfilename = glob(os.path.join(dir, synth_filename2))[0]
            except IndexError:
                if verbose:
                    print('No synthetic file found for data:')
                    print(obs_filename)
                else:
                    pass

    else:
        synth_filename1 = '{}.{}.{}.{}--{}.{}.{}.{}.{}'.format(net1, sta1,
                                                               synth_location,
                                                               cha1, net2,
                                                               sta2,
                                                               synth_location,
                                                               cha2,
                                                               fileformat)

        try:
            sfilename = glob(os.path.join(dir, synth_filename1))[0]

        except IndexError:
            if verbose:
                print('No synthetic file found for data:')
                print(obs_filename)
            else:
                pass

    return sfilename


def glob_obs_corr(sta1, sta2, directory, ignore_network):

    inf1 = sta1.split('.')
    inf2 = sta2.split('.')

    sta1 = inf1[1]
    sta2 = inf2[1]

    cha1 = '??' + inf1[3][-1]
    cha2 = '??' + inf2[3][-1]

    if ignore_network:
        net1 = '*'
        net2 = '*'
    else:
        net1 = inf1[0]
        net2 = inf2[0]

    obs_filename1 = os.path.join(directory, '{}.{}.*.{}*{}.{}.*.{}.*'.format(
        net1, sta1, cha1, net2, sta2, cha2))
    obs_filename2 = os.path.join(directory, '{}.{}.*.{}*{}.{}.*.{}.*'.format(
        net2, sta2, cha2, net1, sta1, cha1))

    if ignore_network:
        obs_files = glob(obs_filename1)
        obs_files.extend(glob(obs_filename2))
    else:
        obs_files = glob(obs_filename1)

    return obs_files
