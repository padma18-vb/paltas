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
numpix = 32

# Define some general image kwargs for the dataset
mask_radius = 0
mag_cut = 0.0

# Define arguments that will be used multiple times
output_ab_zeropoint = 27.79
n_years = 5

# load in focus diverse PSF maps
psf_kernels = np.load('data/psf_images.npy')
psf_kernels=psf_kernels[:, 1, :, :]
psf_kernels[psf_kernels<0] = 0
# normalize psf_kernels to sum to 1
psf_sums = np.sum(psf_kernels,axis=(1,2))
psf_sums = psf_sums.reshape(-1,1,1)
normalized_psfs = psf_kernels/psf_sums

def draw_psf_kernel():
	random_psf_index = np.random.randint(normalized_psfs.shape[0])
	chosen_psf = normalized_psfs[random_psf_index, :, :]
	return chosen_psf


config_dict = {
	'main_deflector':{
		'class': PEMDShear,
		'parameters':{
			'z_lens': norm(loc=0.8, scale=0.6).rvs,
			'gamma': truncnorm(-1.5,1.5,loc=2,scale=0.3),
			'theta_E': truncnorm(-0.5, np.inf, loc=0.8,scale=1),
			'e1': norm(loc=0, scale=0.5),
			'e2': norm(loc=0, scale=0.5,),
			'center_x': 0,
			'center_y': 0,
			'gamma1': norm(loc=0, scale=0.1),
			'gamma2': norm(loc=0, scale=0.1),
			'ra_0':0.0, 'dec_0':0.0
		}
	},
	'lens_light':{
		'class': SingleSersicSource,
		'parameters':{
			'z_source':norm(loc=2, scale=1).rvs,
			'mag_app':norm(loc=20, scale=2.5).rvs,
			'output_ab_zeropoint':output_ab_zeropoint,
			'R_sersic':truncnorm(-0.5, np.inf, loc=0.7,scale=1),
			'n_sersic':norm(4, 0.005),
			'e1':norm(loc=0, scale=0.5),
			'e2':norm(loc=0, scale=0.5),
			'center_x':0,
			'center_y':0
			}
	},
	'source':{
		'class': SingleSersicSource,
		'parameters':{
			'z_source':norm(loc=2, scale=1).rvs,
			'mag_app':norm(loc=24, scale = 2),
			'output_ab_zeropoint':output_ab_zeropoint,
			'R_sersic':truncnorm(-0.5, np.inf, loc=0.7,scale=1),
			'n_sersic':norm(loc=4, scale=0.001),
			'e1':norm(loc=0, scale=0.5),
			'e2':norm(loc=0, scale=0.5),
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
            'mag_app':norm(loc=22, scale=2),
			#'magnitude':truncnorm(-2.0,2.0,loc=-27.42,scale=1.16).rvs,
			'output_ab_zeropoint':output_ab_zeropoint
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
            # ('main_deflector:center_x,lens_light:center_x'):dist.DuplicateScatter(
            #     dist=norm(loc=0,scale=0.07).rvs,scatter=0.005),
            # ('main_deflector:center_y,lens_light:center_y'):dist.DuplicateScatter(
            #     dist=norm(loc=0,scale=0.07).rvs,scatter=0.005),
            ('source:center_x,source:center_y,point_source:x_point_source,'+
                'point_source:y_point_source'):dist.DuplicateXY(
                x_dist=norm(loc=0.0,scale=0.5).rvs,
                y_dist=norm(loc=0.0,scale=0.5).rvs)
		}
	}
}