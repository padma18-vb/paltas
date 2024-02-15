import os

batch_size = 256
# The number of epochs to train for
n_epochs = 30
# The size of the images in the training set
img_size = (33, 33, 1)
# A random seed to us
random_seed = 2
# The list of learning parameters to use
learning_params = ['main_deflector_parameters_theta_E',
	'main_deflector_parameters_gamma1','main_deflector_parameters_gamma2',
	'main_deflector_parameters_gamma','main_deflector_parameters_e1',
	'main_deflector_parameters_e2','main_deflector_parameters_center_x',
	'main_deflector_parameters_center_y']
# Which parameters to consider flipping
flip_pairs = None
# Which terms to reweight
weight_terms = None
# The path to the folder containing the npy images
# for training
npy_folders_train = ['generated_images/no_lens/config_LSST/']
# The path to the tf_record for the training images
tfr_train_paths = [
	os.path.join(path,'data.tfrecord') for path in npy_folders_train]
metadata_paths_train = [
	os.path.join(path,'metadata.csv') for path in npy_folders_train]
# The path to the folder containing the npy images for validation
npy_folder_val = ('generated_images/no_lens_valid/config_LSST/')
# The path to the tf_record for the validation images
tfr_val_path = os.path.join(npy_folder_val,'data.tfrecord')
# The path to the training metadata
# The path to the validation metadata
metadata_path_val = os.path.join(npy_folder_val,'metadata.csv')
# The path to the csv file to read from / write to for normalization
# of learning parameters.
input_norm_path = npy_folders_train[0] + 'norms.csv'
# The detector kwargs to use for on-the-fly noise generation
kwargs_detector = None
# Whether or not to normalize the images by the standard deviation
std_norm_images = True
# A string with which loss function to use.
loss_function = 'diag'
# A string specifying which model to use
model_type = 'xresnet34'
# A string specifying which optimizer to use
optimizer = 'Adam'
# Where to save the	 model weights
model_weights = ('NLNORM/model_weights3/_{epoch:02d}-{val_loss:.2f}.h5')
model_weights_init = ('NLNORM/model_weights2/_49--1.21_last.h5')
# The learning rate for the model
learning_rate = 5e-3*(0.98**(70*1e5/(256*390)))
# Whether or not to use random rotation of the input images
random_rotation = True
# Only train the head
train_only_head = False
# number of steps after which you change the decay rate
steps_per_decay = 390
# csv path
csv_path = '/Users/padmavenkatraman/Documents/StrongLensing/paltas/notebooks/NLNORM/losses3.csv'