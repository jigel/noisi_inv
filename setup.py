#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name = 'noisi',
    version = '0.0.0a0',
    description = 'Package to perform ambient noise source inversions',
    #long_description =
    # url = 
    author = 'J. Igel,L. Ermert,  A. Fichtner',
    author_email  = 'jonas.igel@erdw.ethz.ch, laura.ermert@earth.ox.ac.uk, andreas.fichtner@erdw.ethz.ch',
    # license
    classifiers = [
        'Development Status :: 3 - Alpha',
        'Topic :: Seismology',
        'Programming Language :: Python :: 3',
    ],
    keywords = 'Ambient seismic noise',
    packages = find_packages(),
    package_data={'noisi':['config/config.yml',
                  'config/source_config.yml',
                  'config/measr_config.yml',
                  'config/config_comments.txt',
                  'config/measr_config_comments.txt',
                  'config/source_setup_parameters.yml',
                  'config/stationlist.csv',
                  'config/data_sac_headers.txt',
                  'config/source_config_comments.txt']},
    install_requires = [
        "numpy==1.22.0",
        "scipy",
        "obspy==1.2.2",
        "geographiclib",
        "mpi4py>=2.0.0",
        "geos",
        "proj",
        "pandas",
        "h5py",
        "PyYaml",
        "cartopy",
        "jupyter",
        "pytest",
        "pyasdf",
        "psutil"],
    entry_points = {
        'console_scripts': [
            'noisi = noisi.main:run'
        ]
    },
)

