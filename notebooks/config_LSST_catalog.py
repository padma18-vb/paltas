# Includes a PEMD deflector with external shear, and Sersic sources. 
# Designed to be similar to LSST-like images (though background noise is not yet implemented.)

import numpy as np
from scipy.stats import norm, truncnorm, uniform
import sys
import paltas.Sampling.distributions as dist

from paltas.MainDeflector.simple_deflectors import PEMDShear
from paltas.Sources.sersic import SingleSersicSource
from paltas.PointSource.single_point_source import SinglePointSource
from lenstronomy.Util import kernel_util
from lenstronomy.Util.param_util import phi_q2_ellipticity
import pandas as pd
import os
import paltas

root_path = f'{paltas.__path__[0][:-7]}/notebooks'

# Define the numerics kwargs.
kwargs_numerics = {'supersampling_factor':1}

# This is always the number of pixels for the CCD. If drizzle is used, the
# final image will be larger.
numpix = 33

# Define arguments that will be used multiple times
output_ab_zeropoint = 27
n_years = 5
subtract_lens=True
subtract_source=True
doubles_quads_only=False
catalog = True

# load in data
psf_kernels = np.load(os.path.join(root_path, 'data/norm_resize_psf.npy'), mmap_mode='r+')

def draw_psf_kernel():
	random_psf_index = np.random.randint(psf_kernels.shape[0])
	chosen_psf = psf_kernels[random_psf_index, :, :]
	chosen_psf[chosen_psf<0]=0
	return chosen_psf

index=None

config_dict = {
	'main_deflector':{
		'class': PEMDShear,
		'file': os.path.join(root_path, 'data/deflectors.csv'),
		'parameters':{
			'z_lens': 'ZLENS',
			#'gamma':2,
			'gamma': 'gamma_lens',
			'theta_E': 'EINSTEIN',
			'e1': 'e1_mass', # added to catalog
			'e2': 'e2_mass', # added to catalog
			'center_x': 0, # fixed in OM10
			'center_y': 0, # fixed in OM10
			'gamma1': 'gamma1', # added to catalog
			'gamma2': 'gamma2', # added to catalog
			'ra_0':0.0, 'dec_0':0.0,
		}
	},
	'lens_light':{
		'class': SingleSersicSource,
		'file': os.path.join(root_path, 'data/deflectors.csv'),
		'parameters':{
			'z_source':'ZSRC',
			'mag_app':'APMAG_I', # LENS APPARENT MAG
			'output_ab_zeropoint':output_ab_zeropoint,
			'R_sersic': 'size_true',
			'n_sersic': 'n_sersic',
			'e1': 'e1_light', # added to catalog
			'e2': 'e2_light', # added to catalog
			'center_x':0,
			'center_y':0
			}
	},
	'source':{
		'class': SingleSersicSource,
		'file': os.path.join(root_path, 'data/sources.csv'),
		'parameters':{
			'z_source': 'redshift',
			'mag_app': 'mag_true_i', # SOURCE APPARENT MAG
			'output_ab_zeropoint':output_ab_zeropoint,
			'R_sersic': 'size_true',
			'n_sersic': 'n_sersic',
			'e1':'ellipticity_1_true', # added to catalog
			'e2':'ellipticity_2_true', # added to catalog
			'center_x': 'XSRC',
			'center_y': 'YSRC'
		}
	},
    'point_source':{
		'class': SinglePointSource,
		'file': os.path.join(root_path, 'data/sources.csv'),
		'parameters':{
			'z_source': 'redshift',
            'z_point_source':'ZSRC',
			'x_point_source':'XSRC',
			'y_point_source':'YSRC',
			'mag_app': 'MAGI_IN', # POINT SOURCE APPARENT MAG
			'output_ab_zeropoint':output_ab_zeropoint,
			'compute_time_delays': False
		}
	},
	'cosmology':{
		'file': None,
		'parameters':{
			'cosmology_name': 'planck18'
		}
	},
    'psf':{
		'file': None,

		'parameters':{
			'psf_type':'PIXEL',
			'kernel_point_source':draw_psf_kernel,
			'point_source_supersampling_factor':1
		}
	},

	'lens_equation_solver_parameters':{
		'file': None,
		'solver': 'lenstronomy',
	},

# Currently using the lenstronomy values:
	'detector':{
		'file': None,
		'parameters':{
			'pixel_scale':0.2,'ccd_gain':2.3,'read_noise':10,
			'magnitude_zero_point':output_ab_zeropoint,
			'exposure_time':15,'sky_brightness':20.48,
			'num_exposures':30*n_years,'background_noise':None
		}
	}
}