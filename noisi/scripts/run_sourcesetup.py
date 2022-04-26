"""
Source distribution setup for modelling 

:copyright:
    noisi development team
:license:
    GNU Lesser General Public License, Version 3 and later
    (https://www.gnu.org/copyleft/lesser.html)
"""

import numpy as np
from obspy.signal.invsim import cosine_taper
import h5py
import yaml
import time
from glob import glob
import os
import io
import errno
from noisi import WaveField
from noisi.util.geo import is_land, geographical_distances
from noisi.util.geo import get_spherical_surface_elements
try:
    from noisi.util.plot import plot_grid
    create_plot = True
except ImportError:
    create_plot = False
    pass
import matplotlib.pyplot as plt
from math import pi, sqrt
from warnings import warn
import pprint
from noisi.util.smoothing import get_distance, smooth_gaussian, apply_smoothing_sphere

import functools
print = functools.partial(print, flush=True)


class source_setup(object):

    def __init__(self, args, comm, size, rank):

        if not args.new_model:
            self.setup_source_startingmodel(args,comm,size,rank)
        else:
            self.initialize_source(args)

    def initialize_source(self, args):
        source_model = args.source_model
        project_path = os.path.dirname(source_model)
        noisi_path = os.path.abspath(os.path.dirname(
                                     os.path.dirname(__file__)))
        config_filename = os.path.join(project_path, 'config.yml')

        if not os.path.exists(config_filename):
            raise FileNotFoundError(errno.ENOENT,
                                    os.strerror(errno.ENOENT) +
                                    "\nRun setup_project first.",
                                    config_filename)

        # set up the directory structure:
        if not os.path.isdir(source_model):
            os.mkdir(source_model)
            os.mkdir(os.path.join(source_model, 'observed_correlations'))
            os.mkdir(os.path.join(source_model, 'observed_correlations_slt'))

            for d in ['adjt', 'corr', 'kern','grad']:
                os.makedirs(os.path.join(source_model, 'iteration_0', d))

            # set up the source model configuration file
            with io.open(os.path.join(noisi_path,
                                      'config', 'source_config.yml'), 'r') as fh:
                conf = yaml.safe_load(fh)
                conf['date_created'] = str(time.strftime("%Y.%m.%d"))
                conf['project_name'] = os.path.basename(project_path)
                conf['project_path'] = os.path.abspath(project_path)
                conf['source_name'] = os.path.basename(source_model)
                conf['source_path'] = os.path.abspath(source_model)
                conf['source_setup_file'] = os.path.join(conf['source_path'],
                                                  'source_setup_parameters.yml')

            with io.open(os.path.join(noisi_path,
                                      'config',
                                      'source_config_comments.txt'), 'r') as fh:
                comments = fh.read()

            with io.open(os.path.join(source_model,
                                      'source_config.yml'), 'w') as fh:
                cf = yaml.safe_dump(conf, sort_keys=False, indent=4)
                fh.write(cf)
                fh.write(comments)

            # set up the measurements configuration file
            with io.open(os.path.join(noisi_path,
                                      'config', 'measr_config.yml'), 'r') as fh:
                conf = yaml.safe_load(fh)
                conf['date_created'] = str(time.strftime("%Y.%m.%d"))
            with io.open(os.path.join(noisi_path,
                                      'config',
                                      'measr_config_comments.txt'), 'r') as fh:
                comments = fh.read()

            with io.open(os.path.join(source_model,
                                      'measr_config.yml'), 'w') as fh:
                cf = yaml.safe_dump(conf, sort_keys=False, indent=4)
                fh.write(cf)
                fh.write(comments)

            # set up the measurements configuration file
            with io.open(os.path.join(noisi_path,
                                      'config',
                                      'source_setup_parameters.yml'), 'r') as fh:
                setup = yaml.safe_load(fh)

            with io.open(os.path.join(source_model,
                                      'source_setup_parameters.yml'), 'w') as fh:
                stup = yaml.safe_dump(setup, sort_keys=False, indent=4)
                fh.write(stup)

            os.system('cp ' +
                      os.path.join(noisi_path, 'config', 'stationlist.csv ') +
                      source_model)

            print("Copied default source_config.yml, source_setup_parameters.yml \
    and measr_config.yml to source model directory, please edit and rerun.")
            
        else:
            print(f"Source {source_model} already exists.")
            
        return()

    def setup_source_startingmodel(self, args,comm,size,rank):

        # plotting:
        colors = ['purple', 'g', 'b', 'orange']
        colors_cmaps = [plt.cm.Purples, plt.cm.Greens, plt.cm.Blues,
                        plt.cm.Oranges]
        if rank == 0:
            print("Setting up source starting model.", end="\n")
            
        with io.open(os.path.join(args.source_model,
                                  'source_config.yml'), 'r') as fh:
            source_conf = yaml.safe_load(fh)

        with io.open(os.path.join(source_conf['project_path'],
                                  'config.yml'), 'r') as fh:
            conf = yaml.safe_load(fh)

        with io.open(source_conf['source_setup_file'], 'r') as fh:
            parameter_sets = yaml.safe_load(fh)
            if conf['verbose'] and rank == 0:
                print("The following input parameters are used:", end="\n")
                pp = pprint.PrettyPrinter()
                pp.pprint(parameter_sets)

        # load the source locations of the grid
        grd = np.load(os.path.join(conf['project_path'],
                                   'sourcegrid.npy'))

        # add the voronoi cell surface area or approximate spherical surface elements
        
        
        if 'svp_voronoi_area' in conf and 'svp_grid' in conf or 'auto_data_grid' in conf:
            if conf["svp_voronoi_area"] and conf["svp_grid"] or conf["auto_data_grid"]:
                grd_voronoi = np.load(os.path.join(conf['project_path'],'sourcegrid_voronoi.npy'))
                surf_el = grd_voronoi[2]
                
            else:
                if grd.shape[-1] < 50000:
                    surf_el = get_spherical_surface_elements(grd[0], grd[1])
                else:
                    warn('Large grid; surface element computation slow. Using \
        approximate surface elements.')
                    surf_el = np.ones(grd.shape[-1]) * conf['grid_dx'] ** 2
        else:
            if grd.shape[-1] < 50000:
                surf_el = get_spherical_surface_elements(grd[0], grd[1])
            else:
                warn('Large grid; surface element computation slow. Using \
    approximate surface elements.')
                surf_el = np.ones(grd.shape[-1]) * conf['grid_dx'] ** 2

                
        # get the relevant array sizes
        wfs = glob(os.path.join(conf['project_path'], 'greens', '*.h5'))
        if wfs != []:
            if conf['verbose'] and rank == 0:
                print('Found wavefield stats.')
            else:
                pass
        else:
            raise FileNotFoundError('No wavefield database found. Run \
precompute_wavefield first.')
        with WaveField(wfs[0]) as wf:
            df = wf.stats['Fs']
            n = wf.stats['npad']
        freq = np.fft.rfftfreq(n, d=1. / df)
        n_distr = len(parameter_sets)
        coeffs = np.zeros((grd.shape[-1], n_distr))
        spectra = np.zeros((n_distr, len(freq)))

        # fill in the distributions and the spectra
        for i in range(n_distr):
            
            if parameter_sets[i]['distribution'].endswith('.npy') and rank == 0:                
                coeffs[:, i] = self.distribution_from_data(grd,parameter_sets[i]['distribution'],parameter_sets[i],conf["verbose"])
                
            elif parameter_sets[i]['distribution'].endswith('.h5') and rank == 0:
                coeffs[:, i] = self.distribution_from_prev_model(grd,parameter_sets[i]['distribution'],parameter_sets[i])
                
            elif parameter_sets[i]['distribution'] in ['mfp','matchedfieldprocessing']:
                coeffs[:, i] = self.distribution_from_mfp(grd, args,parameter_sets[i], comm, size, rank)

            else:
                if rank == 0:
                    coeffs[:, i] = self.distribution_from_parameters(grd,
                                                                     parameter_sets[i],
                                                                     conf['verbose'])
                else: 
                    continue
                
            # plot
            outfile = os.path.join(args.source_model,
                                   'source_starting_model_distr%g.png' % i)
            
            if create_plot and rank == 0:
                plot_grid(grd[0], grd[1], coeffs[:, i],
                          outfile=outfile, cmap=colors_cmaps[i%len(colors_cmaps)],
                          sequential=True, normalize=False,
                          quant_unit='Spatial weight (-)',
                          axislabelpad=-0.1,
                          size=10)

            spectra[i, :] = self.spectrum_from_parameters(freq,
                                                          parameter_sets[i])

            
        comm.barrier()
        
        if rank == 0:
            # plotting the spectra
            # plotting is not necessarily done to make sure code runs on clusters
            if create_plot:
                fig1 = plt.figure()
                ax = fig1.add_subplot(1,1,1)
                for i in range(n_distr):
                    ax.plot(freq, spectra[i, :] / spectra.max(),
                            color=colors[i%len(colors_cmaps)])

                ax.set_xlabel('Frequency / Nyquist Frequency')
                plt.xticks([0, freq.max() * 0.25, freq.max() * 0.5,
                           freq.max() * 0.75, freq.max()],
                           ['0', '0.25', '0.5', '0.75', '1'])
                ax.set_ylabel('Rel. PSD norm. to strongest spectrum (-)')
                fig1.savefig(os.path.join(args.source_model,
                                          'source_starting_model_spectra.png'))

            # Save to an hdf5 file
            with h5py.File(os.path.join(args.source_model, 'iteration_0',
                                        'starting_model.h5'), 'w') as fh:
                fh.create_dataset('coordinates', data=grd)
                fh.create_dataset('frequencies', data=freq)
                fh.create_dataset('model', data=coeffs.astype(np.float))
                fh.create_dataset('spectral_basis',
                                  data=spectra.astype(np.float))
                fh.create_dataset('surface_areas',
                                  data=surf_el.astype(np.float))

            # Save to an hdf5 file
            with h5py.File(os.path.join(args.source_model,
                                        'spectral_model.h5'), 'w') as fh:
                uniform_spatial = np.ones(coeffs.shape) * 1.0
                fh.create_dataset('coordinates', data=grd)
                fh.create_dataset('frequencies', data=freq)
                fh.create_dataset('model', data=uniform_spatial.astype(np.float))
                fh.create_dataset('spectral_basis',
                                  data=spectra.astype(np.float))
                fh.create_dataset('surface_areas',
                                  data=surf_el.astype(np.float))
                
        comm.barrier()

    def distribution_from_parameters(self, grd, parameters, verbose=False):

        if parameters['distribution'] == 'homogeneous':
            if verbose:
                print('Adding homogeneous distribution with 1 everywhere')
            distribution = np.ones(grd.shape[-1])
            return(float(parameters['weight']) * distribution)
        
        elif parameters['distribution'] in ['zero','homogeneous_0']:
            if verbose:
                print('Adding homogeneous distribution with 0 everywhere')
            distribution = np.zeros(grd.shape[-1])
            return(float(parameters['weight']) * distribution)
        
        elif parameters['distribution'] == 'random':
            if verbose:
                print('Adding random distribution')
            distribution = np.random.rand(grd.shape[-1])
            return(float(parameters['weight']) * distribution)
        
        elif parameters['distribution'] == 'ocean':
            if verbose:
                print('Adding ocean-only distribution')
            is_ocean = np.abs(is_land(grd[0], grd[1]) - 1.)
            return(float(parameters['weight']) * is_ocean)

        elif parameters['distribution'] == 'gaussian_blob':
            if verbose:
                print('Adding gaussian blob')
                
            
            # implement more than one blob
            if not isinstance(parameters['center_latlon'][0],list):
                n_blobs = 1
            else:
                n_blobs = np.shape(parameters['center_latlon'])[0]
                        
            blob_dist = np.zeros(np.shape(grd)[-1])
                        
            for k in range(n_blobs):
                                    
                # try except in case blob parameters are not in list 
                try: 
                    dist = geographical_distances(grd,
                                          parameters['center_latlon'][k]) / 1000.
                except: 
                    dist = geographical_distances(grd,
                                          parameters['center_latlon']) / 1000.
                    
                try:
                    sigma_km = parameters['sigma_m'][k] / 1000.
                except:
                    sigma_km = parameters['sigma_m'] / 1000.

                blob = np.exp(-(dist ** 2) / (2 * sigma_km ** 2))
                # normalize for a 2-D Gaussian function
                # important: Use sigma in m because the surface elements are in m
                norm_factor = 1. / ((sigma_km * 1000.) ** 2 * 2. * np.pi)
                blob *= norm_factor
                if parameters['normalize_blob_to_unity']:
                    blob /= blob.max()

                if parameters['only_in_the_ocean']:
                    is_ocean = np.abs(is_land(grd[0], grd[1]) - 1.)
                    blob *= is_ocean
                    
                
                blob_dist += blob
                
            blob_dist *= parameters['weight']
                

            return(blob_dist)


    def spectrum_from_parameters(self, freq, parameters):

        mu = parameters['mean_frequency_Hz']
        sig = parameters['standard_deviation_Hz']
        taper = cosine_taper(len(freq), parameters['taper_percent'] / 100.)
        spec = taper * np.exp(-((freq - mu) ** 2) /
                              (2 * sig ** 2))

        if not parameters['normalize_spectrum_to_unity']:
            spec = spec / (sig * sqrt(2. * pi))

        return(spec)
    
    
    def distribution_from_data(self,grd,data,parameters,verbose=False):
        """ 
        Use .npy file to setup source distribution
        Input file has to be: [lat,lon,data] where -90 < lat < 90 and -180 < lon < 180
        """
        
        # need interpolation from data grid to actual grid
        grd_data = np.asarray(np.load(data))
        
        # check if lat and lon are correct        
        if not all(x <= 90 and x >= -90 for x in grd_data[0]):
            raise ValueError("Latitudes for data not within range (-90,90). Exiting..")

        if not all(x <= 180 and x >= -180 for x in grd_data[1]):
            raise ValueError("Longitudes for data not within range (-180,180). Exiting..")
            

        lat_dist = []
        lon_dist = []
        data_dist = []
        
        print("Interpolating data..")
        # nearest neighbour for interpolation
        for k in range(np.size(grd[1])):
            
            dist_var = np.sqrt((grd_data[0]-grd[1][k])**2+(grd_data[1]-grd[0][k])**2)

            # Append interpolated grid to new variables
            lat_dist.append(grd[1][k])
            lon_dist.append(grd[0][k])

            dist_min_idx = np.nanargmin(dist_var)

            # Append to new array
            data_dist.append(grd_data[2][dist_min_idx])

        print('Source distribution setup with data.')
        
        return np.asarray(data_dist)*parameters['weight']
    
    def distribution_from_prev_model(self,grd,model,parameters):
        """ 
        Use .h5 file from previous model
        """
        model_var = h5py.File(model,'r')
        
        model_lat = np.asarray(model_var['coordinates'])[1]
        model_lon = np.asarray(model_var['coordinates'])[0]
        model_data = np.asarray(model_var['model']).T[0]

        grd_data = np.asarray([model_lon,model_lat,model_data])
        
        lat_dist = []
        lon_dist = []
        model_dist = []
        
        print("Interpolating from previous model..")
        # nearest neighbour for interpolation
        for k in range(np.size(grd[1])):
            
            dist_var = np.sqrt((grd_data[0]-grd[0][k])**2+(grd_data[1]-grd[1][k])**2)

            # Append interpolated grid to new variables
            lat_dist.append(grd[1][k])
            lon_dist.append(grd[0][k])

            dist_min_idx = np.nanargmin(dist_var)

            # Append to new array
            model_dist.append(grd_data[2][dist_min_idx])

        print('Source distribution setup with previous model.')
        
        return np.asarray(model_dist)*parameters['weight']
    
    
    def distribution_from_mfp(self,grd,args,parameters,comm,size,rank):
        """
        Use the Matched-Field Processing code to create a starting model from the given correlations
        """
        from noisi.mfp.run_mfp import run_noisi_mfp
        
        # create a config file for the mfp
        # folder structure will be created by mfp code
        if rank == 0:
            print("Source from MFP")
             
            mfp_config = {
                "project_name": f"mfp_startingmodel",
                "output_path": args.project_path,
                "correlation_path": os.path.join(args.source_model,"observed_correlations"),
                "corr_format": ["SAC","sac"],
                "bandpass_filter": args.bandpass[0],
                "stationlist_path": args.stationlist,
                "station_distance_min": 0,
                "station_distance_max": 0,
                "sourcegrid_path": os.path.join(args.project_path,'sourcegrid.npy'),
                "method": ["basic","envelope","envelope_snr","square_envelope_snr"],
                "envelope_snr": 2,
                "stationary_phases": False,            
                "taup_model": "iasp91",
                "phases": [f"{str(args.g_speed).zfill(5)[-4]}.{str(args.g_speed).zfill(5)[-3]}kmps"],
                "phase_pairs": "same",
                "phase_pairs_auto": False,
                "phase_pairs_sum": False,
                "geo_spreading": True,
                "plot": True
            }

            # write mfp config to yaml file

            mfp_config_path = os.path.join(args.project_path,'mfp_config.yml')

            with open(mfp_config_path, 'w') as outfile:
                yaml.safe_dump(mfp_config, outfile, default_flow_style=False)
            
        comm.barrier()
                
        mfp_final = run_noisi_mfp(args,comm,size,rank)
        
        
        
        ## smooth with 2 degrees, copy code from smoothing script
        sigma=[parameters['mfp_smooth']*111000]
        cap=100
        thresh=1e-1000
        
        for ixs in range(len(sigma)):
            sigma[ixs] = float(sigma[ixs])

        coords = [mfp_final[0],mfp_final[1]]
        values = np.array(mfp_final[2]/np.max(np.abs(mfp_final[2])), ndmin=2)

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

            #if rank == 0:
            smoothed_values[i, :] = v
        
        
        
        # normalise
        mfp_smooth = smoothed_values[0]/np.max(np.abs(smoothed_values[0]))
        
            

                
        #return mfp_final[2]/np.max(np.abs(mfp_final[2]))
        
        return mfp_smooth*parameters['weight']
        

    
    
    
