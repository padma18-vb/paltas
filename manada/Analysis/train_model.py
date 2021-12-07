# -*- coding: utf-8 -*-
"""
This script will initialize and train a BNN model on a strong lensing image
dataset.
"""
import argparse, os, sys, glob, math
from importlib import import_module
import tensorflow as tf
from manada.Analysis import dataset_generation, loss_functions, conv_models
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras import optimizers
import pandas as pd


def parse_args():
	"""Parse the input arguments by the user

	Returns:
		(argparse.Namespace): An instance of the Namespace object with the
			users provided values.

	"""
	# Initialize the parser and the possible inputs
	parser = argparse.ArgumentParser()
	parser.add_argument('training_config', help='Path to configuration for '
		+'training the model.')
	parser.add_argument('--tensorboard_dir', default=None, type=str,
		dest='tensorboard_dir', help='Optional path to save a tensorboard ' +
		'output to')
	args = parser.parse_args()
	return args


def main():
	# Get the user provided arguments
	args = parse_args()

	# Get the training parameters from the provided .py file
	config_dir, config_file = os.path.split(os.path.abspath(
		args.training_config))
	sys.path.insert(0, config_dir)
	config_name, _ = os.path.splitext(config_file)
	config_module = import_module(config_name)

	# Grab the arguments passed in by the user
	# The size of each training batch
	batch_size = config_module.batch_size
	# The number of epochs to train for
	n_epochs = config_module.n_epochs
	# The size of the images in the training set
	img_size = config_module.img_size
	# A random seed to us
	random_seed = config_module.random_seed
	# A list of the paths to the TFRecords for the training images
	tfr_train_paths = config_module.tfr_train_paths
	# The path to the TFRecord for the validation images
	tfr_val_path = config_module.tfr_val_path
	# The list of learning parameters to use
	learning_params = config_module.learning_params
	# Which parameters to put in log space
	if hasattr(config_module,'log_learning_params'):
		log_learning_params = config_module.log_learning_params
	else:
		log_learning_params = []
	num_params = len(learning_params+log_learning_params)
	# Which parameters to consider flipping
	if hasattr(config_module,'flip_pairs'):
		flip_pairs = config_module.flip_pairs
	else:
		flip_pairs = None
	# Which parameters to weight
	if hasattr(config_module,'weight_terms'):
		weight_terms = config_module.weight_terms
	else:
		weight_terms = None
	# A list to the paths of the fodlers containing the npy images
	# for training
	npy_folders_train = config_module.npy_folders_train
	# Number of steps per epoch is number of examples over the batch size
	n_npy_files = 0
	for npy_folder in npy_folders_train:
		n_npy_files += len(glob.glob(os.path.join(npy_folder,'image_*.npy')))
	steps_per_epoch = n_npy_files//batch_size
	# The path to the fodler containing the npy images
	# for validation
	npy_folder_val = config_module.npy_folder_val
	# Get the number of validation files as well
	n_val_npy = len(glob.glob(os.path.join(npy_folder_val,'image_*.npy')))
	# A list of the paths to the training metadata
	metadata_paths_train = config_module.metadata_paths_train
	# The path to the validation metadata
	metadata_path_val = config_module.metadata_path_val
	# The path to the csv file to read from / write to for normalization
	# of learning parameters.
	input_norm_path = config_module.input_norm_path
	# The detector kwargs to use for on-the-fly noise generation
	kwargs_detector = config_module.kwargs_detector
	# Whether or not to normalize the images by the standard deviation
	norm_images = config_module.norm_images
	# A string with which loss function to use.
	loss_function = config_module.loss_function
	# A string specifying which model to use
	model_type = config_module.model_type
	# A string specifying which optimizer to use
	optimizer_string = config_module.optimizer
	# Where to save the model weights
	model_weights = config_module.model_weights
	# The learning rate for the model
	learning_rate = config_module.learning_rate
	# Whether or not to apply a random rotation to the input image
	random_rotation = config_module.random_rotation

	# Set the random seed for our network
	tf.random.set_seed(random_seed)

	# Check for tf records for train and validation and prepare them
	# if needed
	print('Checking for training data.')
	for i, tf_path in enumerate(tfr_train_paths):
		if not os.path.exists(tf_path):
			print('Generating new TFRecord at %s'%(tf_path))
			dataset_generation.generate_tf_record(npy_folders_train[i],
				learning_params+log_learning_params,metadata_paths_train[i],
				tf_path)
		else:
			print('TFRecord found at %s'%(tf_path))

	print('Checking for validation data.')
	if not os.path.exists(tfr_val_path):
		print('Generating new TFRecord at %s'%(tfr_val_path))
		dataset_generation.generate_tf_record(npy_folder_val,
			learning_params+log_learning_params,metadata_path_val,tfr_val_path)
	else:
		print('TFRecord found at %s'%(tfr_val_path))

	# Normalize outputs if requested
	if input_norm_path is not None:
		print('Checking for normalization csv')
		metadata = pd.read_csv(metadata_paths_train[0])
		dataset_generation.normalize_outputs(metadata,learning_params,
			input_norm_path,log_learning_params=log_learning_params)

	# If no random rotations are required, the best tool is a tf dataset.
	# Otherwise, it's better to take the output of the tf dataset and then
	# run it through a generator that will do the rotations on the batches
	# for us.
	if random_rotation:
		# Get a generator object that returns rotated images and parameters.
		tf_dataset_t = dataset_generation.generate_rotations_dataset(
			tfr_train_paths,learning_params,batch_size,n_epochs,
			norm_images=norm_images,input_norm_path=input_norm_path,
			kwargs_detector=kwargs_detector,
			log_learning_params=log_learning_params)
	else:
		# Turn our tf records into tf datasets for training and validation
		tf_dataset_t = dataset_generation.generate_tf_dataset(tfr_train_paths,
			learning_params,batch_size,n_epochs,norm_images=norm_images,
			input_norm_path=input_norm_path,kwargs_detector=kwargs_detector,
			log_learning_params=log_learning_params)
	# We shouldn't be adding random noise to validation images. They should
	# be generated with noise
	if kwargs_detector is not None:
		print('Make sure your validation images already have noise! Noise ' +
			'will not be added on the fly for validation.')
	tf_dataset_v = dataset_generation.generate_tf_dataset(tfr_val_path,
		learning_params,min(batch_size,n_val_npy),1,norm_images=norm_images,
		input_norm_path=input_norm_path,kwargs_detector=None,
		log_learning_params=log_learning_params)

	print('Initializing the model')

	# Load the loss function
	if loss_function == 'mse':
		num_outputs = num_params
		loss = loss_functions.MSELoss(num_params,flip_pairs,weight_terms).loss
	elif loss_function == 'diag':
		num_outputs = num_params*2
		loss = loss_functions.DiagonalCovarianceLoss(num_params,
			flip_pairs,weight_terms).loss
	elif loss_function == 'full':
		num_outputs = num_params + int(num_params*(num_params+1)/2)
		loss = loss_functions.FullCovarianceLoss(num_params,flip_pairs,
			weight_terms).loss
	else:
		raise ValueError('%s loss not in the list of supported losses'%(
			loss_function))

	# Load the model
	if model_type == 'resnet50':
		model = conv_models.build_resnet_50(img_size,num_outputs)
	elif model_type == 'xresnet34':
		model = conv_models.build_xresnet34(img_size,num_outputs)
	elif model_type =='alexnet':
		model = conv_models.build_alexnet(img_size,num_outputs)
	else:
		raise ValueError('%s model not in the list of supported models'%(
			model_type))

	# Use learning rate decay for optimal learning
	lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
		learning_rate,decay_steps=steps_per_epoch,decay_rate=0.98,
		staircase=True)

	# Use the desired optimizer
	opt = getattr(optimizers,optimizer_string)(learning_rate=lr_schedule)

	# Compile our model
	model.compile(loss=loss,optimizer=opt,metrics=[loss])

	print('Is model built: ' + str(model.built))

	try:
		model.load_weights(model_weights)
		print('Loaded weights %s'%(model_weights))
	except:
		print('No weights found. Saving new weights to %s'%(model_weights))

	# Set up the callbacks for our training
	callbacks = []
	# If requested, save our model progress to a tensorboard directory
	if args.tensorboard_dir is not None:
		tensorboard = TensorBoard(log_dir=args.tensorboard_dir,
			update_freq='batch')
		callbacks.append(tensorboard)
	# Save the model weights as long as the validation loss is decreasing
	modelcheckpoint = ModelCheckpoint(model_weights,monitor='val_loss',
		save_best_only=True,save_freq='epoch')
	callbacks.append(modelcheckpoint)

	# TODO add validation data.
	model.fit(tf_dataset_t,callbacks=callbacks,epochs=n_epochs,
		steps_per_epoch=steps_per_epoch,validation_data=tf_dataset_v,
		validation_steps=int(math.ceil(n_val_npy/batch_size)))


if __name__ == '__main__':
	main()