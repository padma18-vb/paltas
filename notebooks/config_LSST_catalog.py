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


# Define the numerics kwargs.
kwargs_numerics = {'supersampling_factor':1}

# This is always the number of pixels for the CCD. If drizzle is used, the
# final image will be larger.
numpix = 33

# Define arguments that will be used multiple times
output_ab_zeropoint = 27.79
n_years = 5
catalog = True

# load in data
psf_kernels = np.load('data/norm_resize_psf.npy', mmap_mode='r+')
deflectors = pd.read_csv('data/final_lens_2.csv', index_col=0)
sources = pd.read_csv('data/final_AGN_2.csv', index_col=0)

def draw_psf_kernel():
	random_psf_index = np.random.randint(psf_kernels.shape[0])
	chosen_psf = psf_kernels[random_psf_index, :, :]
	# print(random_psf_index)
	return chosen_psf

index = 3

def phi():
	return deflectors.loc[index,'PHIE']* np.pi / 180
def q_mass():
	return 1/ (1 - deflectors.loc[index,'ELLIP'])
def q_light():
	return (1 - deflectors.loc[index,'ellipticity_true'])/(1 + deflectors.loc[index,'ellipticity_true'])
def q_source_light():	
	return 1 - sources.loc[index,'ellipticity_true']

config_dict = {
	'main_deflector':{
		'class': PEMDShear,
		'parameters':{
			'z_lens': 'ZLENS',
			'gamma': truncnorm(-1.5,1.5,loc=2,scale=0.3).rvs,
			'theta_E': 'EINSTEIN',
			'e1': None, # this is a cross parameter
			'e2': None, # this is a cross parameter
			'center_x': 0, # fixed in OM10
			'center_y': 0, # fixed in OM10
			'gamma1': None, # this is a cross parameter
			'gamma2': None, # this is a cross parameter
			'ra_0':0.0, 'dec_0':0.0
		}
	},
	'lens_light':{
		'class': SingleSersicSource,
		'parameters':{
			'z_source':'ZLENS',
			'mag_app':'APMAG_I',
			'output_ab_zeropoint':output_ab_zeropoint,
			'R_sersic': 'size_true',
			'n_sersic': 'sersic_bulge',
			'e1': None, # this is a cross parameter
			'e2': None, # this is a cross parameter
			'center_x':0,
			'center_y':0
			}
	},
	'source':{
		'class': SingleSersicSource,
		'parameters':{
			'z_source': 'redshift',
			'mag_app': 'mag_true_i',
			'output_ab_zeropoint':output_ab_zeropoint,
			'R_sersic': 'size_true',
			'n_sersic': 'sersic_bulge',
			'e1':None, # this is a cross parameter
			'e2':None, # this is a cross parameter
			'center_x': 'XSRC',
			'center_y': 'YSRC'
		}
	},
    'point_source':{
		'class': SinglePointSource,
		'parameters':{
			'z_source': 'redshift',
            'z_point_source':'ZSRC',
			'x_point_source':'XSRC',
			'y_point_source':'YSRC',
			'mag_abs': 'ABMAGI_IN4',
			'output_ab_zeropoint':output_ab_zeropoint,
			'compute_time_delays': False
		}
	},
	'cosmology':{
		'parameters':{
			'cosmology_name': 'planck18'
		}
	},
    'psf':{
		'parameters':{
			'psf_type':'PIXEL',
			'kernel_point_source':draw_psf_kernel,
			'point_source_supersampling_factor':1
		}
	},

#Currently using the lenstronomy values:
	'detector':{
		'parameters':{
			'pixel_scale':0.2,'ccd_gain':2.3,'read_noise':10,
			'magnitude_zero_point':output_ab_zeropoint,
			'exposure_time':15,'sky_brightness':20.48,
			'num_exposures':30*n_years,'background_noise':None
		}
	},
    'cross_object':{
		'parameters':{
			('main_deflector:e1,main_deflector:e2'):[dist.EllipticitiesTranslation, q_mass, phi],
			('main_deflector:gamma1,main_deflector:gamma2'):[dist.ExternalShearTranslation, 'GAMMA','PHIG'],
			('lens_light:e1,lens_light:e2'): [dist.EllipticitiesTranslation, q_light, phi],
			('source:e1,source:e2'): [dist.EllipticitiesTranslation, q_source_light, 'phi']
		}
	},
	'catalog_path': {
		'parameters': {
			'deflectors': 'data/final_lens_2.csv',
			'sources': 'data/final_AGN_2.csv'
		}
	}
}