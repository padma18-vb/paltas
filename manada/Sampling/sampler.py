# -*- coding: utf-8 -*-
"""
Define the sampler class which reads goes from the distribution dictionary to
drawing samples on the lens parameters.

This module contains the class used to sample parameters for our train and test
set from the input distributions.
"""

# Definte the components we need the sampler to consider.
lensing_components = ['subhalo','los','main_deflector','source','cosmology']


class Sampler():
	"""Class for drawing lens parameter values from input distribution
	dictionaries

	Args:
		configuration_dictionary (dict): An instance of the configuration
			dictionary that will be used to decide how to sample parameter
			values.
	"""

	def __init__(self,configuration_dictionary):
		self.config_dict = configuration_dictionary

	@staticmethod
	def draw_from_dict(draw_dict):
		"""Populates a dict with samples drawn from the specified distributions
		in the input dict.

		Args:
			draw_dict (dict): The dictionary containing keys mapping to values
				or distributions for each parameter.

		Returns:
			(dict): A dict with a drawn value for each parameter.

		Notes:
			Multivariate distribution for parameters should have a key of the
			form 'param_1,param_2,param_3'.
		"""
		param_dict = {}

		# Iterate through the keys in the draw_dict and populate the values of
		# param_dict correctly.
		for key in draw_dict:
			# If the key implies that multiple parameters will be drawn from
			# the distribution, draw the value and then iterate through the
			# parameters.
			if ',' in key:
				# Get the parameters, removing whitespace
				params = key.replace(' ','').split(',')
				# Draw the values
				draw = draw_dict[key]()
				# Check for consistency
				if len(params) != len(draw):
					raise ValueError('Parameters of length %d do'%(len(params)) +
						' not match draw of length %d'%(len(draw)))
				# Populate the keys
				for i, param in enumerate(params):
					param_dict[param] = draw[i]
			# If it's a univariate function just call it.
			elif callable(draw_dict[key]):
				param_dict[key] = draw_dict[key]()
			# If it's a fixed value just populate it.
			else:
				param_dict[key] = draw_dict[key]

		return param_dict

	def sample(self):
		"""Samples from the distributions given in the configuration
		dictionary

		Returns:
			(dict): A dictionary containing the parameter values that were
			sampled.
		"""

		full_param_dict = {}

		# For each possible component of our lensing add the parameters
		for component in lensing_components:
			if component in self.config_dict:
				draw_dict = self.config_dict[component]
				param_dict = self.draw_from_dict(draw_dict)
				full_param_dict[component+'parameters'] = param_dict

		return full_param_dict
