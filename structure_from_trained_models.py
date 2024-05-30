'''
Script to perform the full analysis pipeline around an ensemble of VAEs.
This is tailored to the models that were publicly released with Locatello et al. (2019)
but can be easily extended to your own ensembles.

First, a set of images (the 'fingerprint' set) from the dataset needs to be saved, 
which will be run through every model and every channel in the ensemble.

Then the statistical similarity of the embeddings of the fingerprint set is computed for
every channel, computed as the Bhattacharyya distance.  The exponentiated negative Bhat distance
is the Bhattacharyya coefficient, a value from 0 to 1 where 0 is perfectly distinguishable and 1 is 
perfectly indistinguishable.

Then the Bhat matrices, one per channel, are compared using the measures we discuss in the paper, 
normalized mutual information (NMI) and variation of information (VI).
Finally, using NMI and VI, we visualize structure in the ensemble of channels via the OPTICS algorithm.


Args:

fingerprint_size: This is the number of images to use in the fingerprint. 
	We used 1000 for the results in the paper, but things are much faster for 100 or 300 
	and can be sufficient for preliminary analyses.
dataset: dsprites, smallnorb, or cars3d. 
	We have not looked at the dsprites variants; you'd just need to add code to 
	assemble a fingerprint set of images. You can also (of course) add your own dataset; 
	for these simple architectures it just takes one GPU-day to train an ensemble of 50 models.
outdir: where you want stuff saved.
model_dir: the directory containing the trained models
model_start, model_end: the range of models to include in the analysis. E.g., 0 to 50 would analyze the 50 
	beta-VAEs for dsprites with beta=1.

'''

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os, time
import argparse
import tensorflow_hub as hub
from sklearn import cluster
from matplotlib.gridspec import GridSpec

import utils
import dataset_helper_fns

def get_args():
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('--dataset', default='dsprites',
		choices=['dsprites', 'smallnorb', 'cars3d'])
	parser.add_argument('--outdir', default='.')
	parser.add_argument('--model_dir', default='downloaded_models')
	parser.add_argument('--model_start', type=int, default=0)
	parser.add_argument('--model_end', type=int, default=50)

	parser.add_argument('--fingerprint_size', type=int, default=300)
	args = parser.parse_args()
	return args

def visualize(distance_mat, display_mat, nmi_with_labels, infos, distance_label, out_fname):
	## throw out the low info channels
	info_thresh = 0.01

	nonzero_info_inds = np.where(infos>info_thresh)[0]
	distance_mat = distance_mat[nonzero_info_inds]
	distance_mat = distance_mat[:, nonzero_info_inds]
	display_mat = display_mat[nonzero_info_inds]
	display_mat = display_mat[:, nonzero_info_inds]
	print(f'Discarding channels with less than {info_thresh} bits: {len(nonzero_info_inds)}/{nmi_with_labels.shape[0]} remain')

	OPTICS_MIN_SAMPLES = 20
	clustering = cluster.OPTICS(min_samples=OPTICS_MIN_SAMPLES, metric='precomputed').fit(distance_mat)
	index_reorder = clustering.ordering_
	labels = clustering.labels_[index_reorder]
	reachability = clustering.reachability_[index_reorder]

	fig = plt.figure(figsize=(5, 5))
	gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 6], height_ratios=[1, 6], hspace=0.08, wspace=0.08)

	# Plot the OPTICS reachability profile and any identified groupings
	ax1 = fig.add_subplot(gs[0, 1])
	for cluster_id in np.unique(labels):
		Xk = np.arange(distance_mat.shape[0])[labels == cluster_id]
		Rk = reachability[labels == cluster_id]
		if cluster_id >= 0:
			ax1.axvspan(Xk.min(), Xk.max(), color='k', alpha=0.2)
	ax1.fill_between(np.arange(distance_mat.shape[0]), reachability, color='#454a4a', zorder=100)
	plt.plot(np.arange(distance_mat.shape[0]), reachability, lw=1, color='k', zorder=101)
	ax1.set_xlim(0, len(index_reorder))
	ax1.set_ylim(0, None)
	ax1.set_xticks([])

	# Plot the similarity with the labels
	ax2 = fig.add_subplot(gs[1, 0])
	ax2.plot(nmi_with_labels[nonzero_info_inds][index_reorder], range(len(index_reorder)), lw=1.5)
	ax2.set_ylim(0, len(index_reorder))
	ax2.set_xlim(0, 1)
	ax2.invert_xaxis()
	ax2.invert_yaxis()
	ax2.set_yticks([])

	# Plot the channel similarity matrix.
	axmatrix = fig.add_subplot(gs[1, 1])
	display_mat_organized = display_mat[index_reorder,:]
	display_mat_organized = display_mat_organized[:,index_reorder]

	im = axmatrix.imshow(display_mat_organized, aspect='auto', origin='upper', cmap='magma_r', vmin=0, vmax=1)
	axmatrix.set_xticks([])  
	axmatrix.set_yticks([]) 

	plt.savefig(out_fname)
	plt.show()

def main():
	args = get_args()
	outdir = args.outdir
	if not os.path.exists(outdir):
		os.makedirs(outdir)

	fingerprint_set_fname = os.path.join(outdir, f'{args.dataset}_fingerprint.npz')
	fingerprint_size = args.fingerprint_size
	if not os.path.exists(fingerprint_set_fname):
		## Save size 1000 even if you want to use smaller
		dataset_helper_fns.save_fingerprint_images(args.dataset, fingerprint_set_fname, fingerprint_size=max(fingerprint_size, 1000))

	fingerprint_set = np.load(fingerprint_set_fname, allow_pickle=True)
	fingerprint_set_images = fingerprint_set['images']
	if len(fingerprint_set_images.shape) == 3:
		fingerprint_set_images = np.expand_dims(fingerprint_set_images, -1)
	fingerprint_set_labels = fingerprint_set['labels']

	if fingerprint_set_images.shape[0] < fingerprint_size:
		raise ValueError(f'Saved fingerprint size {fingerprint_set_images.shape[0]} is less than the requested size {fingerprint_size}.  Consider deleting and saving a larger fingerprint.')
	fingerprint_set_images = fingerprint_set_images[:fingerprint_size]
	fingerprint_set_labels = fingerprint_set_labels[:fingerprint_size]

	print(fingerprint_set_images.shape)

	number_repeats = 50
	number_bottleneck_channels = 10
	all_bhat_dist_mats = []
	ct = time.time()
	all_bhat_dist_mats = []
	for model_num in range(args.model_start, args.model_end):
		embed = hub.load(os.path.join(args.model_dir, str(model_num), 'model/tfhub'))
		embs = embed.signatures['gaussian_encoder'](tf.cast(fingerprint_set_images, tf.float32))
		embs_mus = embs['mean']
		embs_logvars = embs['logvar']
		for channel_id in range(number_bottleneck_channels):
			bhat_dist_mat = utils.bhattacharyya_dist_mat(np.reshape(embs_mus[:, channel_id], [fingerprint_size, 1]),
														 np.reshape(embs_logvars[:, channel_id], [fingerprint_size, 1]))
			all_bhat_dist_mats.append(bhat_dist_mat)
	all_bhat_dist_mats = np.stack(all_bhat_dist_mats, 0)

	print(f'{all_bhat_dist_mats.shape} bhat mats computed. Time taken: {(time.time()-ct):.3f} sec.  Computing similarities.')
	ct = time.time()

	pairwise_nmi, pairwise_vi = utils.compute_pairwise_similarities(all_bhat_dist_mats)
	print(f'{all_bhat_dist_mats.shape[0]*(all_bhat_dist_mats.shape[0]+1)//2} similarities computed. Time taken: {(time.time()-ct):.3f} sec.  Visualizing.')

	## Compute the NMI with the labels
	bhat_mats_labels = []
	for label_ind in range(fingerprint_set_labels.shape[-1]):
		label_vec = fingerprint_set_labels[:, label_ind]
		bhat_mat = np.where(np.reshape(label_vec, [-1, 1]) == np.reshape(label_vec, [1, -1]), 1, 0)
		bhat_mats_labels.append(bhat_mat)
		
	## Kind of clunky, but we need to convert these to distance matrices first, that give approx 0s and 1s when exponentiated
	bhat_dist_mats_labels = -np.log(np.clip(bhat_mats_labels, 1e-10, None))

	pairwise_nmi_with_labels, pairwise_vi_with_labels = utils.compute_pairwise_similarities(all_bhat_dist_mats, bhat_dist_mats_labels)
	## Get the info per channel so we can detect and discard those below a threshold
	infos = -np.mean(np.log2(np.mean(np.exp(-all_bhat_dist_mats), axis=-1)), axis=-1)
	for (distance_label, distance_mat, display_mat) in zip(
			['VI', 'NMI'],
			[np.clip(pairwise_vi, 0, None), -np.log(np.clip(pairwise_nmi, 1e-4, None))],
			[np.exp(-pairwise_vi), pairwise_nmi]
			):
		visualization_fname = os.path.join(outdir, f'structure_{args.dataset}_{args.model_start}to{args.model_end}using{distance_label}_.png')
		visualize(distance_mat, display_mat, pairwise_nmi_with_labels, infos, distance_label, visualization_fname)

if __name__ == '__main__':
	main()
