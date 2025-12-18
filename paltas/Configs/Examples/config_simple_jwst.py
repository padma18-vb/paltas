# Includes a PEMD deflector with external shear, and Sersic sources. Includes 
# a simple observational effect model that roughly matches HST effects for
# Wide Field Camera 3 (WFC3) IR channel with the F160W filter.

import numpy as np
from scipy.stats import norm, truncnorm, uniform
from paltas.MainDeflector.simple_deflectors import PEMDShear
from paltas.Sources.sersic import SingleSersicSource

import sys
import paltas.Sampling.distributions as dist

from paltas.PointSource.single_point_source import SinglePointSource
from lenstronomy.Util import kernel_util
# Define the numerics kwargs.
kwargs_numerics = {'supersampling_factor':1}

# This is always the number of pixels for the CCD. If drizzle is used, the
# final image will be larger.
numpix = 128
catalog=False

# Define some general image kwargs for the dataset

# Define arguments that will be used multiple times
output_ab_zeropoint = 28.17

config_dict = {
	'main_deflector':{
		'class': PEMDShear,
		'parameters':{
			'z_lens': truncnorm(-0.5, 0.5, loc=0.8,scale=1).rvs,
			'gamma': norm(loc=2.0,scale=0.2).rvs,
			'theta_E': truncnorm(-0.5, np.inf, loc=0.8,scale=1).rvs,
			'e1':norm(loc=0,scale=0.2).rvs,
			'e2':norm(loc=0,scale=0.2).rvs,
			'center_x': None,
			'center_y': None,
			'gamma1': norm(loc=0,scale=0.1).rvs,
			'gamma2': norm(loc=0,scale=0.1).rvs,
			'ra_0':0.0, 'dec_0':0.0
		}
	},
	'lens_light':{
		'class': SingleSersicSource,
		'parameters':{
			'z_source':1.7,
			'mag_app':norm(loc=20.5, scale=2).rvs,
			'output_ab_zeropoint':output_ab_zeropoint,
			'R_sersic':truncnorm(-0.5, np.inf, loc=0.7,scale=1).rvs,
			'n_sersic':norm(loc=4, scale=0.005).rvs,
			'e1':None,
			'e2':None,
			'center_x':None,
			'center_y':None
			}
	},
	'source':{
		'class': SingleSersicSource,
		'parameters':{
			'z_source':1.7,
			'mag_app':norm(loc=24, scale = 2).rvs,
			'output_ab_zeropoint':output_ab_zeropoint,
			'R_sersic':truncnorm(-0.5, np.inf, loc=0.7,scale=1).rvs, # maybe this should be smaller
			'n_sersic':norm(loc=4, scale=0.001).rvs,
			'e1':norm(loc=0, scale=0.1).rvs,
			'e2':norm(loc=0, scale=0.1).rvs,
			'center_x':norm(loc=0.0,scale=0.4).rvs,
			'center_y':norm(loc=0.0,scale=0.4).rvs,
            'gamma1': norm(loc=0, scale=0.1).rvs,
            'gamma2': norm(loc=0, scale=0.1).rvs}
    },
    # 'point_source':{
	# 	'class': SinglePointSource,
	# 	'parameters':{
    #         'z_point_source':None,
	# 		'x_point_source':None,
	# 		'y_point_source':None,
    #         # range: 19 to 25
    #         'mag_app':norm(loc=22, scale=2).rvs,
	# 		#'magnitude':truncnorm(-2.0,2.0,loc=-27.42,scale=1.16).rvs,
	# 		'mag_pert':dist.MultipleValues(dist=truncnorm(-1/0.3,np.inf,1,0.3).rvs,num=10),
	# 		'output_ab_zeropoint':output_ab_zeropoint,
	# 		'compute_time_delays': False
	# 	}
	# },
	'cosmology':{
		'parameters':{
			'cosmology_name': 'planck18'
		}
	},
	'psf':{
		'parameters':{
			'psf_type':'GAUSSIAN',
			'fwhm': 0.03
		}
	},
	'detector':{
		'parameters':{
			'pixel_scale':0.040,'ccd_gain':1.58,'read_noise':3.0,
			'magnitude_zero_point':output_ab_zeropoint,
			'exposure_time':1380,'sky_brightness':21.83,
			'num_exposures':4,'background_noise':None
		}
	},
    'cross_object':{
		'parameters':{
            ('main_deflector:center_x,lens_light:center_x'):dist.DuplicateScatter(
                dist=norm(loc=0,scale=0.06).rvs,scatter=0.001),
            ('main_deflector:center_y,lens_light:center_y'):dist.DuplicateScatter(
                dist=norm(loc=0,scale=0.06).rvs,scatter=0.001),
			('main_deflector:e1,lens_light:e1'):dist.DuplicateScatter(
                dist=norm(loc=0,scale=0.2).rvs,scatter=0.12),
            ('main_deflector:e2,lens_light:e2'):dist.DuplicateScatter(
                dist=norm(loc=0,scale=0.2).rvs,scatter=0.12)
			# ('main_deflector:z_lens,lens_light:z_source,source:z_source'): dist.RedshiftsPointSource(
			# 	z_lens_min=0,z_lens_mean=0.5,z_lens_std=0.6,
			# 	z_source_min=0,z_source_mean=2,z_source_std=0.6)
		}
	}
}

