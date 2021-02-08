from noisi.mfp.mfp_code.util.source_grid_svp import svp_grid


def create_sourcegrid(svp_grid_config):
    """
    Function to create source grid from input arguments
    """
    
    # turn mfp args into values
    sigma = svp_grid_config['svp_sigma']
    beta = svp_grid_config['svp_beta']
    phi_min = svp_grid_config['svp_phi_min']
    phi_max = svp_grid_config['svp_phi_max']
    lat_0 = svp_grid_config['svp_lat_0']
    lon_0 = svp_grid_config['svp_lon_0']
    gamma = svp_grid_config['svp_gamma']
    plot = svp_grid_config['svp_plot']
    dense_antipole = svp_grid_config['svp_dense_antipole']
    only_ocean = svp_grid_config['svp_only_ocean']
    
    sourcegrid = svp_grid(sigma,beta,phi_min,phi_max,lat_0,lon_0,gamma,plot,dense_antipole,only_ocean)
    
    return sourcegrid