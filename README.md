## Noisi - ambient noise cross-correlation modeling and inversion

This tool can be used to simulate noise cross-correlations and sensitivity kernels to noise sources.

### Installation requirements

Install requirements (easiest done with anaconda)
- [obspy](https://docs.obspy.org/) 
- PyYaml
- pandas
- mpi4py
- proj
- geos
- geographiclib
- cartopy
- h5py
- jupyter
- pytest
- pyasdf
- psutil

Additionally, install [instaseis](http://instaseis.net/), if you plan to use it for Green's functions (`conda install instaseis`).

Install jupyter notebook if you intend to run the tutorial (see below).

If you encounter problems with mpi4py, try removing it and reinstalling it using pip (`pip install mpi4py`).

### Installation step-by-step

Clone the repository with git:  
`git clone https://github.com/jigel/noisi_inv.git`

Create a new environment and activate it:  
`conda create -n noisi_inv`  
`conda activate noisi_inv`

Install a few necessary packages:  
`conda install proj geos mpi4py h5py instaseis --yes`  

Install noisi_inv (in the noisi_inv directory) which should install all other packages.
Change into the `noisi_inv/` directory. Call `pip install .` here, or call `pip install -v -e .` if you intend to modify the code.

After installation, change to the `noisi_inv/noisi` directory and run `pytest`. If you encounter any errors (warnings are o.k.), we'll be grateful if you let us know. 

### Getting started

Noisi_inv is slightly different to noisi as it includes a complete inversion workflow. This is all setup in the run_inversion.py file. 
To do a little test inversion and check if it was installed correctly you can run the following command:  
`python run_inversion.py inversion_config.yml`


To see an overview of the tool, type `noisi --help`.
A step-by-step tutorial for jupyter notebook can be found in the `noisi_inv/noisi` directory.
Examples on how to set up an inversion and how to import a wavefield from axisem3d are found in the noisi/examples directory.

### Tutorial: Inversion setup
We have added a jupyter notebook to help you setup a config file for an inversion. All the different parameters are explained within that notebook. It is also available as python script. 

It is recommended to download a pre-computed wavefield if you want to do proper inversions (http://ds.iris.edu/ds/products/syngine/). These are then easily implemented in the config file. 


