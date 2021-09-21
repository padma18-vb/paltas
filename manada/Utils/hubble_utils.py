# -*- coding: utf-8 -*-
"""
Utility functions for producing Hubble drizzled images.
"""
import copy
import numpy as np
from astropy.io import fits
from astropy.wcs import wcs
from drizzle.drizzle import Drizzle
from scipy.interpolate import RectBivariateSpline
import warnings

WCS_ORIGIN = 1


def offset_wcs(w,pixel_offset):
	"""Return wcs w shifted by pixel_offset pixels.

	Args:
		w (astropy.wcs.wcs.WCS): An istance of the WCS class that will
			be copied and offset.
		pixel_offset (tuple): The x,y offset to apply in pixel units.

	Returns:
		(astropy.wcs.wcs.WCS): The offset WCS object.
	"""
	# Copy the object and add the pixel offset to the reference pixel
	# position.
	w_out = copy.deepcopy(w)
	w_out.wcs.crpix += np.array(pixel_offset)
	return w_out


def distort_image(img_high_res,w_high_res,w_dither,offset_pattern):
	"""Create distorted Hubble image in _flt frame

	Args:
		img_high_res (np.array): A high resolution image that will be
			downsampled to produce each of the dithered images.
		w_high_res (astropy.wcs.wcs.WCS): The WCS class for the high resolution
			image. This should not include any distortion effects.
		w_dither (astropy.wcs.wcs.WCS): The WCS class for the dithered image.
			This should have the same pixel scale as the camera and should
			be centered at the same point as w_high_res. The offsets will be
			accounted for in the function call.
		offset_pattern ([tuple,...]): A list of tuples, each containing an x,y
			pair of offsets in pixel coordinates.

	Returns:
		(np.array): A len(offset_pattern) x dither_image.shape sized numpy
			array for each of the dithered images.
	"""
	# First create the interpolator for our image in the sky. Values are assumed
	# to be measured at the center of pixels.
	x_interp = np.arange(img_high_res.shape[0])+0.5
	y_interp = np.arange(img_high_res.shape[1])+0.5
	# Interpolation will be linear.
	img_itp = RectBivariateSpline(x_interp, y_interp, img_high_res,
		kx=1,ky=1)

	# Create the coordinates we want to plot the dithered image on.
	dith_shape = w_dither._naxis
	x_dith,y_dith = np.meshgrid(np.arange(dith_shape[0])+0.5,
		np.arange(dith_shape[1])+0.5,indexing='ij')

	# Array in which we'll store the images
	img_dither_array = np.zeros((len(offset_pattern),)+tuple(dith_shape))

	# Now for each offset requested, generate the dithered image.
	for oi,offset in enumerate(offset_pattern):
		# Calculate the offset WCS
		w_offset = offset_wcs(w_dither,offset)

		# Use that to calculate the ra and dec of each pixel and map
		# that to the image values from the interpolation
		ra_off,dec_off = w_offset.all_pix2world(x_dith,y_dith,
			WCS_ORIGIN)
		x_dith_int,y_dith_int = w_high_res.all_world2pix(ra_off,dec_off,
			WCS_ORIGIN)

		img_dither_array[oi] += img_itp(x_dith_int,y_dith_int,grid=False)

	# Multiply by scaling since we want sum not mean.
	img_dither_array *= w_dither.wcs.cdelt[0]/w_high_res.wcs.cdelt[0]
	img_dither_array *= w_dither.wcs.cdelt[1]/w_high_res.wcs.cdelt[1]

	return img_dither_array


def generate_downsampled_wcs(high_res_shape,high_res_pixel_scale,
		low_res_pixel_scale,wcs_distortion):
	"""Generate a wcs mapping onto the specified lower resolution. If
	specified, geometric distortions will be included in the wcs object.

	Args:
		high_res_shape ([int,..]): The shape of the high resolution image
		high_res_pixel_scale (float): The pixel width of the high
			resolution image in units of arcseconds.
		low_res_pixel_scale (float): The pixel width of the lower
			resolution target image in units of arcseconds.
		wcs_distortion (astropy.wcs.wcs.WCS): An instance of the
			WCS class that includes the desired gemoetric distortions as
			SIP coefficients. Note that the parameters relating to
			pixel scale, size of the image, and reference pixels will be
			overwritten. If None no gemoetric distortions will be applied.

	Returns:
		astropy.wcs.wcs.WCS: The wcs object corresponding to the
			downsampled image.
	"""
	# If wcs distortion was specified, use it to seed the fits file we
	# will later use to create the wcs object. This is a little
	# circular, but it's difficult to modify a wcs once it's been
	# created.
	if wcs_distortion is not None:
		warnings.warn('wcs distortion code is not tested.')
		hdul = wcs_distortion.to_fits()
	else:
		hdul = fits.HDUList([fits.PrimaryHDU()])

	# Get the shape of the expected data
	high_res_shape = list(high_res_shape)
	scaling = low_res_pixel_scale/high_res_pixel_scale
	low_res_shape = copy.copy(high_res_shape)
	low_res_shape[0] = low_res_shape[0]//scaling
	low_res_shape[1] = low_res_shape[1]//scaling

	# Modify the fits header to match our desired low resolution
	# configuration.
	hdul[0].header['WCSAXES'] = 2
	hdul[0].header['NAXIS1'] = low_res_shape[0]
	hdul[0].header['NAXIS2'] = low_res_shape[1]
	hdul[0].header['CRPIX1'] = (low_res_shape[0]/2,
		'Pixel coordinate of reference point')
	hdul[0].header['CRPIX2'] = (low_res_shape[1]/2,
		'Pixel coordinate of reference point')
	hdul[0].header['CDELT1'] = (low_res_pixel_scale/3600,
		'[deg] Coordinate increment at reference point')
	hdul[0].header['CDELT2'] = (low_res_pixel_scale/3600,
		'[deg] Coordinate increment at reference point')
	hdul[0].header['CUNIT1'] = ('deg',
		'Units of coordinate increment and value ')
	hdul[0].header['CUNIT2'] = ('deg',
		'Units of coordinate increment and value ')
	if wcs_distortion:
		hdul[0].header['CTYPE1'] = 'RA-TAN-SIP'
		hdul[0].header['CTYPE2'] = 'DEC-TAN-SIP'
	else:
		hdul[0].header['CTYPE1'] = 'RA'
		hdul[0].header['CTYPE2'] = 'DEC'
	hdul[0].header['CRVAL1'] = (20,
		'[deg] Coordinate value at reference point')
	hdul[0].header['CRVAL2'] = (-70,
		'[deg] Coordinate value at reference point')

	# Use the first object to generate our WCS object
	return wcs.WCS(fobj=hdul,header=hdul[0].header)


def hubblify(img_high_res,high_res_pixel_scale,detector_pixel_scale,
	drizzle_pixel_scale,noise_model,psf_model,offset_pattern,
	wcs_distortion=None):
	"""Generates a simulated drizzled HST image, accounting for
	gemoetric distortions, the dithering pattern, and correlated
	read noise from drizzling.

	Args:
		img_high_res (np.array): A high resolution image that will be
			downsampled to produce each of the dithered images.
		high_res_pixel_scale (float): The pixel width of the
			high resolution image in units of arcseconds.
		detector_pixel_scale (float): The pixel width of the detector
			in units of arcseconds.
		drizzle_pixel_scale (float): The pixel width of the final
			drizzled product in units of arcseconds.
		noise_model (function): A function that maps from an input
			numpy array of the image to a realization of the noise. This
			should operate on the degraded images.
		psf_model (function): A function that maps from an input image
			to an output psf-convovled image. This should operate on the
			degraded images.
		offset_pattern ([tuple,...]): A list of x,y coordinate pairs
			specifying the offset of each dithered image from the coordinate
			frame used to generate the high resolution image. Specifying shifts
			that place the degraded image outside of the high resolution image
			will cause interpolation errors.
		wcs_distortion (astropy.wcs.wcs.WCS): An instance of the
			WCS class that includes the desired gemoetric distortions.
			Note that the parameters relating to pixel scale, size of
			the image, and reference pixels will be overwritten. If None
			no gemoetric distortions will be applied.

	Returns:
		(np.array): The drizzled image produced from len(offset_pattern)
			number of dithered exposures of img_high_res. Note that this
			means that the drizzled image will have the noise statistics and
			flux of len(offset_pattern) combined exposures.
	"""
	# Create our three base WCS systems, one for the input high res, one for the
	# input highres, one for the drizzled image, and one for the final dithered
	# image.
	w_high_res = generate_downsampled_wcs(img_high_res.shape,
		high_res_pixel_scale,high_res_pixel_scale,None)
	w_driz = generate_downsampled_wcs(img_high_res.shape,high_res_pixel_scale,
		drizzle_pixel_scale,None)
	w_dither = generate_downsampled_wcs(img_high_res.shape,
		high_res_pixel_scale,detector_pixel_scale,wcs_distortion)

	# Initialize our drizzle class with the target output wcs.
	driz = Drizzle(outwcs=w_driz)

	# Get the distorted sub images to which we will add noise and then add
	# them to our drizzle.
	img_dither_array = distort_image(img_high_res,w_high_res,w_dither,
		offset_pattern)

	for d_i,image_dither in enumerate(img_dither_array):

		# Add the psf and the noise to each dithered image.
		image_dither = psf_model(image_dither)
		image_dither += noise_model(image_dither)

		# Feed to drizzle, note WCS translation
		driz.add_image(image_dither,offset_wcs(w_dither, offset_pattern[d_i]))

	# Get final image from drizzle
	return driz.outsci
