'''
Just grab a random set of images and whatever labels go with those images, and save them
for fingerprinting every channel in every model of an ensemble.

This code is an ugly mishmash, but it just copies the necessary parts from disentanglement-lib so that the images
for cars3d and smallnorb are the exact same as what was trained on.
Copied from the corresponding files in 
https://github.com/google-research/disentanglement_lib/tree/master/disentanglement_lib/data/ground_truth

dsprites is simple enough to trust tensorflow_datasets, and its code can be easily modified for whatever
dataset you want to use
'''

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import os
import PIL

def save_fingerprint_images(dataset_name, out_fname, data_dir='.', fingerprint_size=1000):
	####################################################################################################################
	if dataset_name == 'dsprites':
		import tensorflow_datasets as tfds
		dataset_labels = ['label_orientation', 'label_shape', 'label_scale', 'label_x_position', 'label_y_position']
		dset, dset_info = tfds.load(dataset_name, data_dir=data_dir, with_info=True, decoders={
		    'image': tfds.decode.SkipDecoding(),
		})

		dset_train = dset['train']
		dset_train = dset_train.map(lambda example: (example['image'], tf.stack([example[label] for label in dataset_labels], 0)))

		dset_train = dset_train.shuffle(1_000_000)

		dset_train = dset_train.map(
		    lambda image, labels: (dset_info.features['image'].decode_example(image), labels))
		dset_train = dset_train.map(lambda image, labels: (tf.image.convert_image_dtype(image, tf.float32)*255., labels))

		fingerprint_set = next(iter(dset_train.batch(fingerprint_size)))
		np.savez(out_fname,
		      images=fingerprint_set[0],
		      labels=fingerprint_set[1]
		      )
		print(f'Saved fingerprint set for {dataset_name} to {out_fname}')
		return
	####################################################################################################################
	elif dataset_name == 'smallnorb':
		SMALLNORB_TEMPLATE = os.path.join(
		    "Data/disentanglement_lib", "small_norb",
		    "smallnorb-{}-{}.mat")

		SMALLNORB_CHUNKS = [
		    "5x46789x9x18x6x2x96x96-training",
		    "5x01235x9x18x6x2x96x96-testing",
		]

		def _load_small_norb_chunks(path_template, chunk_names):
		  """Loads several chunks of the small norb data set for final use."""
		  list_of_images, list_of_features = _load_chunks(path_template, chunk_names)
		  features = np.concatenate(list_of_features, axis=0)
		  features[:, 3] = features[:, 3] / 2  # azimuth values are 0, 2, 4, ..., 24
		  return np.concatenate(list_of_images, axis=0), features


		def _load_chunks(path_template, chunk_names):
		  """Loads several chunks of the small norb data set into lists."""
		  list_of_images = []
		  list_of_features = []
		  for chunk_name in chunk_names:
		    norb = _read_binary_matrix(path_template.format(chunk_name, "dat"))
		    list_of_images.append(_resize_images(norb[:, 0]))
		    norb_class = _read_binary_matrix(path_template.format(chunk_name, "cat"))
		    norb_info = _read_binary_matrix(path_template.format(chunk_name, "info"))
		    list_of_features.append(np.column_stack((norb_class, norb_info)))
		  return list_of_images, list_of_features


		def _read_binary_matrix(filename):
		  """Reads and returns binary formatted matrix stored in filename."""
		  with tf.io.gfile.GFile(filename, "rb") as f:
		    s = f.read()
		    magic = int(np.frombuffer(s, "int32", 1))
		    ndim = int(np.frombuffer(s, "int32", 1, 4))
		    eff_dim = max(3, ndim)
		    raw_dims = np.frombuffer(s, "int32", eff_dim, 8)
		    dims = []
		    for i in range(0, ndim):
		      dims.append(raw_dims[i])

		    dtype_map = {
		        507333717: "int8",
		        507333716: "int32",
		        507333713: "float",
		        507333715: "double"
		    }
		    data = np.frombuffer(s, dtype_map[magic], offset=8 + eff_dim * 4)
		  data = data.reshape(tuple(dims))
		  return data

		def _resize_images(integer_images):
		  resized_images = np.zeros((integer_images.shape[0], 64, 64))
		  for i in range(integer_images.shape[0]):
		    image = PIL.Image.fromarray(integer_images[i, :, :])
		    image = image.resize((64, 64), PIL.Image.ANTIALIAS)
		    resized_images[i, :, :] = image
		  return resized_images / 255.

		images, features = _load_small_norb_chunks(SMALLNORB_TEMPLATE,
		                                           SMALLNORB_CHUNKS)
		factor_sizes = [5, 10, 9, 18, 6]
		# Instances are not part of the latent space.
		latent_factor_indices = [0, 2, 3, 4]
		num_total_factors = features.shape[1]

		random_inds = np.random.choice(images.shape[0], size=fingerprint_size)
		fingerprint_set_images = images[random_inds]
		fingerprint_set_labels = features[random_inds]

		np.savez(out_fname,
		        images=fingerprint_set_images,
		        labels=fingerprint_set_labels
		        )
		print(f'Saved fingerprint set for {dataset_name} to {out_fname}')
		return
	####################################################################################################################
	elif dataset_name == 'cars3d':
		import scipy.io as sio
		from sklearn.utils import extmath
		"""Cars3D data set.

		The data set was first used in the paper "Deep Visual Analogy-Making"
		(https://papers.nips.cc/paper/5845-deep-visual-analogy-making) and can be
		downloaded from http://www.scottreed.info/. The images are rescaled to 64x64.

		The ground-truth factors of variation are:
		0 - elevation (4 different values)
		1 - azimuth (24 different values)
		2 - object type (183 different values)
		"""

		class StateSpaceAtomIndex(object):
		  """Index mapping from features to positions of state space atoms."""

		  def __init__(self, factor_sizes, features):
		    """Creates the StateSpaceAtomIndex.

		    Args:
		      factor_sizes: List of integers with the number of distinct values for each
		        of the factors.
		      features: Numpy matrix where each row contains a different factor
		        configuration. The matrix needs to cover the whole state space.
		    """
		    self.factor_sizes = factor_sizes
		    num_total_atoms = np.prod(self.factor_sizes)
		    self.factor_bases = num_total_atoms / np.cumprod(self.factor_sizes)
		    feature_state_space_index = self._features_to_state_space_index(features)
		    if np.unique(feature_state_space_index).size != num_total_atoms:
		      raise ValueError("Features matrix does not cover the whole state space.")
		    lookup_table = np.zeros(num_total_atoms, dtype=np.int64)
		    lookup_table[feature_state_space_index] = np.arange(num_total_atoms)
		    self.state_space_to_save_space_index = lookup_table

		  def features_to_index(self, features):
		    """Returns the indices in the input space for given factor configurations.

		    Args:
		      features: Numpy matrix where each row contains a different factor
		        configuration for which the indices in the input space should be
		        returned.
		    """
		    state_space_index = self._features_to_state_space_index(features)
		    return self.state_space_to_save_space_index[state_space_index]

		  def _features_to_state_space_index(self, features):
		    """Returns the indices in the atom space for given factor configurations.

		    Args:
		      features: Numpy matrix where each row contains a different factor
		        configuration for which the indices in the atom space should be
		        returned.
		    """
		    if (np.any(features > np.expand_dims(self.factor_sizes, 0)) or
		        np.any(features < 0)):
		      raise ValueError("Feature indices have to be within [0, factor_size-1]!")
		    return np.array(np.dot(features, self.factor_bases), dtype=np.int64)

		def _load_data():
		  dataset = np.zeros((24 * 4 * 183, 64, 64, 3))
		  all_files = [x for x in tf.io.gfile.listdir(CARS3D_PATH) if ".mat" in x]
		  for i, filename in enumerate(all_files):
		    data_mesh = _load_mesh(filename)
		    factor1 = np.array(list(range(4)))
		    factor2 = np.array(list(range(24)))
		    all_factors = np.transpose([
		        np.tile(factor1, len(factor2)),
		        np.repeat(factor2, len(factor1)),
		        np.tile(i,
		                len(factor1) * len(factor2))
		    ])
		    indexes = index.features_to_index(all_factors)
		    dataset[indexes] = data_mesh
		  return dataset


		def _load_mesh(filename):
		  """Parses a single source file and rescales contained images."""
		  with open(os.path.join(CARS3D_PATH, filename), "rb") as f:
		    mesh = np.einsum("abcde->deabc", sio.loadmat(f)["im"])
		  flattened_mesh = mesh.reshape((-1,) + mesh.shape[2:])
		  rescaled_mesh = np.zeros((flattened_mesh.shape[0], 64, 64, 3))
		  for i in range(flattened_mesh.shape[0]):
		    pic = PIL.Image.fromarray(flattened_mesh[i, :, :, :])
		    # pic.thumbnail((64, 64, 3), PIL.Image.ANTIALIAS)
		    pic = pic.resize((64, 64), PIL.Image.ANTIALIAS)
		    rescaled_mesh[i, :, :, :] = np.array(pic)
		  return rescaled_mesh * 1. / 255


		factor_sizes = [4, 24, 183]

		latent_factor_indices = [0, 1, 2]

		features = extmath.cartesian(
		        [np.array(list(range(i))) for i in factor_sizes])
		index = StateSpaceAtomIndex(factor_sizes, features)

		data_shape = [64, 64, 3]
		images = _load_data()

		random_inds = np.random.choice(images.shape[0], size=fingerprint_size)
		fingerprint_set_images = images[random_inds]
		fingerprint_set_labels = features[random_inds]

		np.savez(out_fname,
		        images=fingerprint_set_images,
		        labels=fingerprint_set_labels
		        )
		print(f'Saved fingerprint set for {dataset_name} to {out_fname}')
		return
	else:
		raise ValueError(f'{dataset_name} not implemented for fingerprinting.')
