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
multiband = True
filter_list = ['F090W', 'F115W', 'F150W', 'F200W', 'F356W','F444W']
filter_dependent_properties = ['lens_light_parameters_mag_app',
                               'lens_light_parameters_R_sersic', 
                               'lens_light_parameters_output_ab_zeropoint', 
                               'source_parameters_R_sersic',
                               'source_parameters_mag_app',
                               'source_parameters_output_ab_zeropoint', 
                               'psf_parameters_fwhm', 
                               'detector_parameters_pixel_scale',
                               'detector_parameters_magnitude_zero_point',
                               'detector_parameters_sky_brightness']

catalog=False
### add in all jaguar broadband -- make sure all the jades medium bands are included
### initially test on cosmos filters 
JWST_FILTERS = {
    "F090W": {"pixel_scale": 0.031, "fwhm": 0.032, "zp": 28.02, "sky": 26.0},
    "F115W": {"pixel_scale": 0.031, "fwhm": 0.038, "zp": 28.04, "sky": 25.8},
    "F150W": {"pixel_scale": 0.031, "fwhm": 0.048, "zp": 28.05, "sky": 25.5},
    "F200W": {"pixel_scale": 0.031, "fwhm": 0.063, "zp": 27.99, "sky": 25.0},
    "F356W": {"pixel_scale": 0.063, "fwhm": 0.11,  "zp": 27.78, "sky": 23.5},
    "F444W": {"pixel_scale": 0.063, "fwhm": 0.15,  "zp": 27.70, "sky": 23.0},
}
chosen_filter = 'F356W'

# Define some general image kwargs for the dataset

# Define arguments that will be used multiple times

config_dict = {
	'main_deflector':{
		'class': PEMDShear,
		'parameters':{
			'z_lens': None,
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
			'z_source':None,
			'mag_app':{"F090W":norm(loc=21, scale = 2).rvs,
              "F115W":norm(loc=21, scale = 2).rvs, 
              "F150W":norm(loc=21, scale = 2).rvs,
              "F200W":norm(loc=21, scale = 2).rvs,
              "F356W":norm(loc=21, scale = 2).rvs, 
              "F444W":norm(loc=21, scale=2).rvs},
			'output_ab_zeropoint':{'F090W': JWST_FILTERS['F090W']['zp'],
                          "F115W":JWST_FILTERS['F115W']['zp'], 
						"F150W":JWST_FILTERS['F150W']['zp'],
						"F200W":JWST_FILTERS['F200W']['zp'],
						"F356W":JWST_FILTERS['F356W']['zp'], 
						"F444W":JWST_FILTERS['F444W']['zp']},
            # {'g':truncnorm(-2,2,loc=0.35,scale=0.05).rvs,'r':truncnorm(-2,2,loc=0.35,scale=0.05).rvs},
			'R_sersic':{"F090W":truncnorm(-0.5, np.inf, loc=0.5,scale=1).rvs,
               "F115W":truncnorm(-0.5, np.inf, loc=0.55,scale=1).rvs,
               "F150W":truncnorm(-0.5, np.inf, loc=0.6,scale=1).rvs,
               "F200W":truncnorm(-0.5, np.inf, loc=0.65,scale=1).rvs,
               "F356W":truncnorm(-0.5, np.inf, loc=0.7,scale=1).rvs,
               "F444W":truncnorm(-0.5, np.inf, loc=0.75,scale=1).rvs},
            # 'R_sersic': truncnorm(-0.5, np.inf, loc=0.7,scale=1).rvs,
			'n_sersic':norm(loc=4, scale=0.005).rvs,
			'e1,e2':dist.EllipticitiesTranslation(
				q_dist=truncnorm(-np.inf,1.,loc=0.85,scale=0.15).rvs,
				phi_dist=uniform(loc=-np.pi/2,scale=np.pi).rvs),
			'center_x':None,
			'center_y':None
			}
	},
	'source':{
		'class': SingleSersicSource,
		'parameters':{
			'z_source':None,
			'mag_app':{"F090W":norm(loc=21, scale = 2).rvs, 
              "F115W":norm(loc=22, scale=2).rvs,
              "F150W":norm(loc=21, scale = 2).rvs, 
              "F200W":norm(loc=22, scale=2).rvs,
              "F356W":norm(loc=21, scale = 2).rvs, 
              "F444W":norm(loc=21, scale=2).rvs},
			'output_ab_zeropoint':{'F090W': JWST_FILTERS['F090W']['zp'],
                          "F115W":JWST_FILTERS['F115W']['zp'], 
						"F150W":JWST_FILTERS['F150W']['zp'],
						"F200W":JWST_FILTERS['F200W']['zp'],
						"F356W":JWST_FILTERS['F356W']['zp'], 
						"F444W":JWST_FILTERS['F444W']['zp']},
			'R_sersic':{"F090W":truncnorm(-2,2,loc=0.35,scale=0.05).rvs,
               "F115W":truncnorm(-2,2,loc=0.35,scale=0.05).rvs,
               "F150W":truncnorm(-2,2,loc=0.35,scale=0.05).rvs,
               "F200W":truncnorm(-2,2,loc=0.35,scale=0.05).rvs,
               "F356W":truncnorm(-2,2,loc=0.35,scale=0.05).rvs,
               "F444W":truncnorm(-2,2,loc=0.35,scale=0.05).rvs,}, # maybe this should be smaller
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
			'fwhm': {'F090W': JWST_FILTERS['F090W']['fwhm'],
                          "F115W":JWST_FILTERS['F115W']['fwhm'], 
						"F150W":JWST_FILTERS['F150W']['fwhm'],
						"F200W":JWST_FILTERS['F200W']['fwhm'],
						"F356W":JWST_FILTERS['F356W']['fwhm'], 
						"F444W":JWST_FILTERS['F444W']['fwhm']}
		}
	},
    # what should the exposure time be?
    # check that noise is being added appropriately
	'detector':{
		'parameters':{
			'pixel_scale':{'F090W': JWST_FILTERS['F090W']['pixel_scale'],
                          "F115W":JWST_FILTERS['F115W']['pixel_scale'], 
						"F150W":JWST_FILTERS['F150W']['pixel_scale'],
						"F200W":JWST_FILTERS['F200W']['pixel_scale'],
						"F356W":JWST_FILTERS['F356W']['pixel_scale'], 
						"F444W":JWST_FILTERS['F444W']['pixel_scale']},'ccd_gain':1.0,'read_noise':5.0,
			'magnitude_zero_point':{'F090W': JWST_FILTERS['F090W']['zp'],
                          "F115W":JWST_FILTERS['F115W']['zp'], 
						"F150W":JWST_FILTERS['F150W']['zp'],
						"F200W":JWST_FILTERS['F200W']['zp'],
						"F356W":JWST_FILTERS['F356W']['zp'], 
						"F444W":JWST_FILTERS['F444W']['zp']},
			'exposure_time': 3600 * 2,'sky_brightness':{'F090W': JWST_FILTERS['F090W']['sky'],
                          "F115W":JWST_FILTERS['F115W']['sky'], 
						"F150W":JWST_FILTERS['F150W']['sky'],
						"F200W":JWST_FILTERS['F200W']['sky'],
						"F356W":JWST_FILTERS['F356W']['sky'], 
						"F444W":JWST_FILTERS['F444W']['sky']},
			'num_exposures':4,'background_noise':None
		}
	},
    'cross_object':{
		'parameters':{
            ('main_deflector:center_x,lens_light:center_x'):dist.DuplicateScatter(
                dist=norm(loc=0,scale=0.06).rvs,scatter=0.001),
            ('main_deflector:center_y,lens_light:center_y'):dist.DuplicateScatter(
                dist=norm(loc=0,scale=0.06).rvs,scatter=0.001),
			# ('main_deflector:e1,lens_light:e1'):dist.DuplicateScatter(
            #     dist=norm(loc=0,scale=0.2).rvs,scatter=0.12),
            # ('main_deflector:e2,lens_light:e2'):dist.DuplicateScatter(
            #     dist=norm(loc=0,scale=0.2).rvs,scatter=0.12)
			('main_deflector:z_lens,lens_light:z_source,source:z_source'): dist.RedshiftsLensLight(
				z_lens_min=0,z_lens_mean=0.5,z_lens_std=0.6,
				z_source_min=0,z_source_mean=2,z_source_std=0.6)
		}
	}
}

