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

# Define the numerics kwargs.
kwargs_numerics = {'supersampling_factor':1}

# This is always the number of pixels for the CCD. If drizzle is used, the
# final image will be larger.
numpix = 33

# Define arguments that will be used multiple times
output_ab_zeropoint = 27
n_years = 1

# load in focus diverse PSF maps
catalog=False
index=None
double_quads_only=True
psf_kernels = np.load('data/norm_resize_psf.npy', mmap_mode='r+')

def draw_psf_kernel():
	random_psf_index = np.random.randint(psf_kernels.shape[0])
	chosen_psf = psf_kernels[random_psf_index, :, :]
	chosen_psf[chosen_psf<0]=0
	return chosen_psf


config_dict = {
	'main_deflector':{
		'class': PEMDShear,
		'parameters':{
			'z_lens': None,
			'gamma': truncnorm(-1.5,1.5,loc=2,scale=0.3).rvs,
			'theta_E': truncnorm(-0.5, np.inf, loc=0.8,scale=1).rvs,
			'e1': norm(loc=0, scale=0.1).rvs,
			'e2': norm(loc=0, scale=0.1).rvs,
			'center_x': None,
			'center_y': None,
			'gamma1': norm(loc=0, scale=0.1).rvs,
			'gamma2': norm(loc=0, scale=0.1).rvs,
			'ra_0':0.0, 'dec_0':0.0
		}
	},
	'lens_light':{
		'class': SingleSersicSource,
		'parameters':{
			'z_source':None,
			'mag_app':norm(loc=20.5, scale=2).rvs,
			'output_ab_zeropoint':output_ab_zeropoint,
			'R_sersic':truncnorm(-0.5, np.inf, loc=0.7,scale=1).rvs,
			'n_sersic':norm(loc=4, scale=0.005).rvs,
			'e1':norm(loc=0, scale=0.1).rvs,
			'e2':norm(loc=0, scale=0.1).rvs,
			'center_x':None,
			'center_y':None
			}
	},
	'source':{
		'class': SingleSersicSource,
		'parameters':{
			'z_source':None,
			'mag_app':norm(loc=24, scale = 2).rvs,
			'output_ab_zeropoint':output_ab_zeropoint,
			'R_sersic':truncnorm(-0.5, np.inf, loc=0.7,scale=1).rvs,
			'n_sersic':norm(loc=4, scale=0.001).rvs,
			'e1':norm(loc=0, scale=0.1).rvs,
			'e2':norm(loc=0, scale=0.1).rvs,
			'center_x':None,
			'center_y':None
		}
	},
    'point_source':{
		'class': SinglePointSource,
		'parameters':{
            'z_point_source':None,
			'x_point_source':None,
			'y_point_source':None,
            # range: 19 to 25
            'mag_app':norm(loc=22, scale=2).rvs,
			#'magnitude':truncnorm(-2.0,2.0,loc=-27.42,scale=1.16).rvs,
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
            ('main_deflector:center_x,lens_light:center_x'):dist.DuplicateScatter(
                dist=norm(loc=0,scale=0.05).rvs,scatter=0.001),
            ('main_deflector:center_y,lens_light:center_y'):dist.DuplicateScatter(
                dist=norm(loc=0,scale=0.05).rvs,scatter=0.001),
            ('source:center_x,source:center_y,point_source:x_point_source,'+
                'point_source:y_point_source'):dist.DuplicateXY(
                x_dist=norm(loc=0.0,scale=0.4).rvs,
                y_dist=norm(loc=0.0,scale=0.4).rvs),
			('main_deflector:z_lens,lens_light:z_source,source:z_source,'+ 
				'point_source:z_point_source'): dist.RedshiftsPointSource(
				z_lens_min=0,z_lens_mean=0.5,z_lens_std=0.6,
				z_source_min=0,z_source_mean=2,z_source_std=0.6)
		}
	}
}