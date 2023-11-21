# Configuration for training of CNN for STRIDES30 test

import numpy as np
from scipy.stats import norm, uniform, truncnorm, randint
import paltas.Sampling.distributions as dist
from paltas.MainDeflector.simple_deflectors import PEMDShear
from paltas.Sources.sersic import SingleSersicSource
from paltas.PointSource.single_point_source import SinglePointSource

# lsst stats: https://smtn-002.lsst.io/
# lsst key numbers: https://www.lsst.org/scientists/keynumbers

# zero point of telescope - lowest magnitude 
# ‘instrumental zeropoint’ in each bandpass, the AB magnitude which would produce one count per second

output_ab_zeropoint = 27.79

lsst_camera = {"read_noise": 10,  # will be <10
          "pixel_scale": 0.2,
          "ccd_gain": 2.3,
        }

i_band_obs = {
    "exposure_time": 15.0,
    "sky_brightness": 20.48,
    "magnitude_zero_point": 27.79,
    "num_exposures": 460,
    "seeing": 0.71,
    "psf_type": "GAUSSIAN",
}

kwargs_numerics = {'supersampling_factor':1}

# size of cutout
numpix = 80 # numpix * pixel_scale = how many arcseconds in the sky

# quads_only
#doubles_quads_only = True
# point source magnification cut
#ps_magnification_cut = 2

# load in a PSF kernel
from astropy.io import fits
from lenstronomy.Util import kernel_util


config_dict = {
	'main_deflector':{
		'class': PEMDShear,
		'parameters':{
			'z_lens': truncnorm(-2.5,np.inf,loc=0.5,scale=0.25).rvs,
			'gamma': truncnorm(-10.,np.inf,loc=2.0,scale=0.2).rvs,
            # switch to a more uniform Einstein Radius dist.
            'theta_E': truncnorm(-4.,np.inf,loc=0.8,scale=0.2).rvs,
            # truncated to fit in 80x80 pixels
			#'theta_E': truncnorm(-6.0,4.5,loc=1.1,scale=0.1).rvs,
            'e1':norm(loc=0,scale=0.24).rvs,
            'e2':norm(loc=0,scale=0.24).rvs,
			#'e1,e2': dist.EllipticitiesTranslation(
			#	q_dist=truncnorm(-5./3.,1.,loc=0.7,scale=0.3).rvs,
			#	phi_dist=uniform(loc=-np.pi/2,scale=np.pi).rvs),
			'center_x':None,
			'center_y':None,
			'gamma1':norm(loc=0,scale=0.16).rvs,
            'gamma2':norm(loc=0,scale=0.16).rvs,
            #'gamma1,gamma2': dist.ExternalShearTranslation(
            #    gamma_dist=truncnorm(0.,5./3.,loc=0.,scale=0.3).rvs,
			#	phi_dist=uniform(loc=-np.pi/2,scale=np.pi).rvs),
			'ra_0':0.0,
			'dec_0':0.0,
		}
	},
    'source':{
		'class': SingleSersicSource,
		'parameters':{
			'z_source':truncnorm(-5,np.inf,loc=2.,scale=0.4).rvs,
            # range: 20 to 27, centered at 23.5
            'mag_app':truncnorm(-3./2.,3./2.,loc=25.6,scale=1.1).rvs,
			#'magnitude':truncnorm(-2.0,2.0,loc=-26.17,scale=1.70).rvs,
			'output_ab_zeropoint':output_ab_zeropoint,
			'R_sersic':truncnorm(-(5./8.),np.inf,loc=0.5,scale=0.8).rvs,
			'n_sersic':truncnorm(-1.25,np.inf,loc=3.,scale=2.).rvs,
			'e1':truncnorm(-2.5,2.5,loc=0,scale=0.24).rvs,
            'e2':truncnorm(-2.5,2.5,loc=0,scale=0.24).rvs,
			'center_x':None,
			'center_y':None}

	},
    'lens_light':{
		'class': SingleSersicSource,
		'parameters':{
			'z_source':None,
            # range: 17 to 23
            'mag_app':truncnorm(-3./2.,3./2.,loc=50,scale=2.).rvs,
			#'magnitude':truncnorm(-1.5,2.0,loc=-23.10,scale=1.72).rvs,
			'output_ab_zeropoint':output_ab_zeropoint,
			'R_sersic':truncnorm(-(1./.8),np.inf,loc=1.0,scale=0.8).rvs,
			'n_sersic':truncnorm(-1.25,np.inf,loc=3.,scale=2.).rvs,
			'e1':truncnorm(-2.5,2.5,loc=0,scale=0.24).rvs,
            'e2':truncnorm(-2.5,2.5,loc=0,scale=0.24).rvs,
			'center_x':None,
			'center_y':None}
	},
    'point_source':{
		'class': SinglePointSource,
		'parameters':{
            'z_point_source':None,
			'x_point_source':None,
			'y_point_source':None,
            # range: 19 to 25
            'mag_app':truncnorm(-3./2.,3./2.,loc=25.,scale=2.).rvs,
			#'magnitude':truncnorm(-2.0,2.0,loc=-27.42,scale=1.16).rvs,
			'output_ab_zeropoint':output_ab_zeropoint,
			'mag_pert': dist.MultipleValues(dist=truncnorm(-1/0.3,np.inf,1,0.3).rvs,num=10),
            'compute_time_delays':False
		}
	},
    'cosmology':{
		'parameters':{
			'cosmology_name': 'planck18'
		}
	},
    'psf':{
		'parameters':{
            'psf_type': 'GAUSSIAN',
            'fwhm': 0.9
		}
	},
    
	'detector':{
		'parameters':{
			'pixel_scale':lsst_camera['pixel_scale'],'ccd_gain':lsst_camera['ccd_gain'],'read_noise':lsst_camera['read_noise'],
			'magnitude_zero_point':output_ab_zeropoint,
			'exposure_time':15.,'sky_brightness':i_band_obs['sky_brightness'],
			'num_exposures':i_band_obs['num_exposures'],'background_noise':None
		}
	},

    'cross_object':{
		'parameters':{
            ('main_deflector:center_x,lens_light:center_x'):dist.DuplicateScatter(
                dist=norm(loc=0,scale=0.07).rvs,scatter=0.005),
            ('main_deflector:center_y,lens_light:center_y'):dist.DuplicateScatter(
                dist=norm(loc=0,scale=0.07).rvs,scatter=0.005),
            ('source:center_x,source:center_y,point_source:x_point_source,'+
                'point_source:y_point_source'):dist.DuplicateXY(x_dist=uniform(loc=-0.2,scale=0.4).rvs,
                y_dist=uniform(loc=-0.2,scale=0.4).rvs),
			('main_deflector:z_lens,lens_light:z_source,source:z_source,'+
                 'point_source:z_point_source'):dist.RedshiftsPointSource(
				z_lens_min=0,z_lens_mean=0.5,z_lens_std=0.2,
				z_source_min=0,z_source_mean=2,z_source_std=0.4)
		}
	}
}
    
