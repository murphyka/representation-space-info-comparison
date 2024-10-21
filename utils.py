import numpy as np
import tensorflow as tf 


def bhattacharyya_dist_mat(mus, logvars):
  """Computes Bhattacharyya distances between multivariate Gaussians.
  The Bhattacharyya coefficient is the exponentiated negative distance.
  Args:
    mus: [N, d] float array of the means of the Gaussians.
    logvars: [N, d] float array of the log variances of the Gaussians (so we're assuming diagonal
    covariance matrices; these are the logs of the diagonal).
  Returns:
    [N, N] array of distances.
  """
  N = mus.shape[0]
  embedding_dimension = mus.shape[1]

  ## Manually broadcast
  mus1 = np.tile(mus[:, np.newaxis], [1, N, 1])
  logvars1 = np.tile(logvars[:, np.newaxis], [1, N, 1])
  mus2 = np.tile(mus[np.newaxis], [N, 1, 1])
  logvars2 = np.tile(logvars[np.newaxis], [N, 1, 1])
  difference_mus = mus1 - mus2  # [N, M, embedding_dimension]; we want [N, N, embedding_dimension, 1]
  difference_mus = difference_mus[..., np.newaxis]
  difference_mus_T = np.transpose(difference_mus, [0, 1, 3, 2])

  sigma_diag = 0.5 * (np.exp(logvars1) + np.exp(logvars2))  ## [N, N, embedding_dimension], but we want a diag mat [N, N, embedding_dimension, embedding_dimension]
  sigma_mat = np.expand_dims(sigma_diag, -1) * np.expand_dims(np.ones_like(sigma_diag), -2) * np.reshape(np.eye(embedding_dimension), [1, 1, embedding_dimension, embedding_dimension])
  sigma_mat_inv = np.expand_dims(1./sigma_diag, -1) * np.expand_dims(np.ones_like(sigma_diag), -2) * np.reshape(np.eye(embedding_dimension), [1, 1, embedding_dimension, embedding_dimension])

  log_determinant_sigma = np.sum(np.log(sigma_diag), axis=-1)
  log_determinant_sigma1 = np.sum(logvars1, axis=-1)
  log_determinant_sigma2 = np.sum(logvars2, axis=-1)
  term1 = 0.125 * (difference_mus_T @ sigma_mat_inv @ difference_mus).reshape([N, N])
  term2 = 0.5 * (log_determinant_sigma - 0.5 * (log_determinant_sigma1  + log_determinant_sigma2))
  return term1+term2

@tf.function(experimental_relax_shapes=True)
def bhattacharyya_dist_mat_tf(mus, logvars):
  """Computes Bhattacharyya distances between multivariate Gaussians.
  Args:
    mus: [N, d] float array of the means of the Gaussians.
    logvars: [N, d] float array of the log variances of the Gaussians (so we're assuming diagonal
    covariance matrices; these are the logs of the diagonal).
  Returns:
    [N, N] array of distances.
  """
  N = tf.shape(mus)[0]
  embedding_dimension = tf.shape(mus)[1]

  mus = tf.cast(mus, tf.float64)
  logvars = tf.cast(logvars, tf.float64)

  ## Manually broadcast in case either M or N is 1
  mus1 = tf.tile(tf.expand_dims(mus, 1), [1, N, 1])
  logvars1 = tf.tile(tf.expand_dims(logvars, 1), [1, N, 1])
  mus2 = tf.tile(tf.expand_dims(mus, 0), [N, 1, 1])
  logvars2 = tf.tile(tf.expand_dims(logvars, 0), [N, 1, 1])
  difference_mus = mus1 - mus2  # [N, M, embedding_dimension]; we want [N, M, embedding_dimension, 1]
  difference_mus = tf.expand_dims(difference_mus, -1)
  difference_mus_T = tf.transpose(difference_mus, [0, 1, 3, 2])

  sigma_diag = 0.5 * (tf.exp(logvars1) + tf.exp(logvars2))  ## [N, M, embedding_dimension], but we want a diag mat [N, M, embedding_dimension, embedding_dimension]
  # sigma_mat = np.apply_along_axis(np.diag, -1, sigma_diag)
  sigma_mat = tf.expand_dims(sigma_diag, -1) * tf.expand_dims(tf.ones_like(sigma_diag, dtype=tf.float64), -2) * tf.reshape(tf.eye(embedding_dimension, dtype=tf.float64), [1, 1, embedding_dimension, embedding_dimension])
  # sigma_mat_inv = np.apply_along_axis(np.diag, -1, 1./sigma_diag)
  sigma_mat_inv = tf.expand_dims(1./sigma_diag, -1) * tf.expand_dims(tf.ones_like(sigma_diag, dtype=tf.float64), -2) * tf.reshape(tf.eye(embedding_dimension, dtype=tf.float64), [1, 1, embedding_dimension, embedding_dimension])

  log_determinant_sigma = tf.reduce_sum(tf.math.log(sigma_diag), axis=-1)
  log_determinant_sigma1 = tf.reduce_sum(logvars1, axis=-1)
  log_determinant_sigma2 = tf.reduce_sum(logvars2, axis=-1)
  term1 = 0.125 * tf.reshape(difference_mus_T @ sigma_mat_inv @ difference_mus, [N, N])
  term2 = 0.5 * (log_determinant_sigma - 0.5 * (log_determinant_sigma1 + log_determinant_sigma2))
  return term1+term2

def compute_pairwise_similarities(bhat_distance_mats, bhat_distance_mats2=None):
  """Computes the full set of pairwise VI and NMI comparisons given a list of Bhattacharyya distance matrices.
  Requires that the distance matrices were computed with the same set of data points, in the same order. 

  Args:
    bhat_distance_mats: length M list of [N, N] float arrays of the pairwise Bhattacharyya distances between N data points. 
  Returns:
    [M, M] array of pairwise VI, [M, M] array of pairwise NMI 
  """
  if bhat_distance_mats2 is None:
    bhat_distance_mats2 = bhat_distance_mats
    symmetric_mat = 1  ## use this bit to skip half of the computations
  else:
    symmetric_mat = 0

  pairwise_nmi = np.zeros((bhat_distance_mats.shape[0], bhat_distance_mats2.shape[0]))
  pairwise_vi = np.zeros((bhat_distance_mats.shape[0], bhat_distance_mats2.shape[0]))

  # Compute I(X;U) with the lower bound from Kolchinsky + Tracey 2017
  infos1 = -np.mean(np.log2(np.mean(np.exp(-bhat_distance_mats), axis=-1)), axis=-1)
  infos2 = -np.mean(np.log2(np.mean(np.exp(-bhat_distance_mats2), axis=-1)), axis=-1)
  # Compute I(X;U,U`) as the info about the dataset if you get a message from the same channel U twice
  infos2x1 = -np.mean(np.log2(np.mean(np.exp(-bhat_distance_mats*2), axis=-1)), axis=-1)
  infos2x2 = -np.mean(np.log2(np.mean(np.exp(-bhat_distance_mats2*2), axis=-1)), axis=-1)
  for ind1 in range(bhat_distance_mats.shape[0]):
    for ind2 in range(ind1*symmetric_mat, bhat_distance_mats2.shape[0]):
      info1 = infos1[ind1]
      info2 = infos2[ind2]
      info11 = infos2x1[ind1]
      info22 = infos2x2[ind2]
      info12 = -np.mean(np.log2(np.mean(np.exp(-(bhat_distance_mats[ind1]+bhat_distance_mats2[ind2])), axis=-1)), axis=-1)

      nmi = (info1+info2-info12) / np.sqrt((2*info1-info11)*(2*info2-info22))
      vi = info12*2 - info11 - info22

      pairwise_nmi[ind1, ind2] = nmi
      pairwise_vi[ind1, ind2] = vi

      if symmetric_mat:
        pairwise_nmi[ind2, ind1] = nmi
        pairwise_vi[ind2, ind1] = vi

  return pairwise_nmi, pairwise_vi

@tf.function(experimental_relax_shapes=True)
def compute_likelihoods_mc(samples, mus, logvars, diag=False):
  """Computes likelihoods of samples u given a set of posterior parameters mus and logvars.

  Args:
    samples: [N, d] float64 tensor of points in the d-dimensional latent space. 
    mus: [M, d] float tensor of the means of M posteriors. 
    logvars: [M, d] float tensor of the log variances of M posteriors. 
    diag: Whether there is a 1:1 correspondence between the samples and the posteriors,
      and you want to get the likelihood under the posterior (True) or under the aggregated posterior (False)
  Returns:
    [N] float tensor of the likelihoods of the samples. 
  """
  mus = tf.cast(mus, tf.float64)
  logvars = tf.cast(logvars, tf.float64)
  sample_size = tf.shape(samples)[0]
  evaluation_batch_size = tf.shape(mus)[0]
  embedding_dimension = tf.shape(mus)[-1]
  stddevs = tf.exp(logvars/2.)
  # Expand dimensions to broadcast and compute the pairwise distances between
  # the sampled points and the centers of the conditional distributions
  samples = tf.reshape(samples,
    [sample_size, 1, embedding_dimension])
  mus = tf.reshape(mus, [1, evaluation_batch_size, embedding_dimension])
  distances_ui_muj = samples - mus

  normalized_distances_ui_muj = distances_ui_muj / tf.reshape(stddevs, [1, evaluation_batch_size, embedding_dimension])
  p_ui_cond_xj = tf.exp(-tf.reduce_sum(normalized_distances_ui_muj**2, axis=-1)/2. - \
    tf.reshape(tf.reduce_sum(logvars, axis=-1), [1, evaluation_batch_size])/2.)
  normalization_factor = (2.*np.pi)**(tf.cast(embedding_dimension, tf.float64)/2.)
  p_ui_cond_xj = p_ui_cond_xj / normalization_factor
  if diag:
    return tf.linalg.diag_part(p_ui_cond_xj)
  else:
    return tf.reduce_sum(p_ui_cond_xj, axis=-1)


def monte_carlo_info(mus, logvars, number_random_samples=20000):
  """Estimates the information I(U;X) transmitted about a dataset by the complete list of posteriors.

  Args:
    mus: [M, d] float array of the means of M posteriors, where M is the length of the dataset
    logvars: [M, d] float array of the log variances of M posteriors. 
    number_random_samples: the number of Monte Carlo samples
  Returns:
    The estimate of the mutual information, in bits, and its standard error.
  """
  sample_size = 2000  ## How many samples to evaluate at a time
  chunk_eval_size = 10_000  ## How many data points to evaluate at a time; the involved matrices will be [sample_size, chunk_eval_size]
  info_estimates = []
  emb_dim = mus.shape[-1]
  for rand_sample in range(number_random_samples//sample_size):
    rand_inds = np.random.choice(mus.shape[0], size=sample_size)
    rand_sample = tf.random.normal(shape=(sample_size, emb_dim),
                                  mean=mus[rand_inds],
                                  stddev=tf.exp(logvars[rand_inds]/2.))
    rand_sample = tf.cast(rand_sample, tf.float64)
    posterior_probs = compute_likelihoods_mc(rand_sample, mus[rand_inds], logvars[rand_inds], diag=True)
    marginal_probs = np.zeros((sample_size))
    for start_ind in range(0, mus.shape[0], chunk_eval_size):
      end_ind = min(start_ind+chunk_eval_size, mus.shape[0])
      marginal_probs = marginal_probs + compute_likelihoods(rand_sample, mus[start_ind:end_ind], logvars[start_ind:end_ind])
    marginal_probs = marginal_probs / mus.shape[0]

    info_estimates.append(tf.math.log(posterior_probs/marginal_probs))
  info_estimates = np.array(info_estimates)/np.log(2)
  return np.mean(info_estimates), np.std(info_estimates)/np.sqrt((number_random_samples//sample_size)*sample_size)
