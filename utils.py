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

  determinant_sigma = np.prod(sigma_diag, axis=-1)
  determinant_sigma1 = np.exp(np.sum(logvars1, axis=-1))
  determinant_sigma2 = np.exp(np.sum(logvars2, axis=-1))
  term1 = 0.125 * (difference_mus_T @ sigma_mat_inv @ difference_mus).reshape([N, N])
  term2 = 0.5 * np.log(determinant_sigma / np.sqrt(determinant_sigma1 * determinant_sigma2))
  return term1+term2

@tf.function
def bhattacharyya_dist_mat_tf(mus, logvars):
  """Tensorflow version of the bhat computation (for speed when optimizing)
  Args:
    mus: [N, d] float array of the means of the Gaussians.
    logvars: [N, d] float array of the log variances of the Gaussians (so we're assuming diagonal
    covariance matrices; these are the logs of the diagonal).
  Returns:
    [N, N] array of distances.
  """
  N = tf.shape(mus)[0]
  embedding_dimension = tf.shape(mus)[1]

  ## Manually broadcast
  mus1 = tf.tile(tf.expand_dims(mus, 1), [1, N, 1])
  logvars1 = tf.tile(tf.expand_dims(logvars, 1), [1, N, 1])
  mus2 = tf.tile(tf.expand_dims(mus, 0), [N, 1, 1])
  logvars2 = tf.tile(tf.expand_dims(logvars, 0), [N, 1, 1])
  difference_mus = mus1 - mus2  # [N, N, embedding_dimension]; we want [N, N, embedding_dimension, 1]
  difference_mus = tf.expand_dims(difference_mus, -1)
  difference_mus_T = tf.transpose(difference_mus, [0, 1, 3, 2])

  sigma_diag = 0.5 * (tf.exp(logvars1) + tf.exp(logvars2))  ## [N, N, embedding_dimension], but we want a diag mat [N, N, embedding_dimension, embedding_dimension]
  sigma_mat = tf.expand_dims(sigma_diag, -1) * tf.expand_dims(tf.ones_like(sigma_diag), -2) * tf.reshape(tf.eye(embedding_dimension), [1, 1, embedding_dimension, embedding_dimension])
  sigma_mat_inv = tf.expand_dims(1./sigma_diag, -1) * tf.expand_dims(tf.ones_like(sigma_diag), -2) * tf.reshape(tf.eye(embedding_dimension), [1, 1, embedding_dimension, embedding_dimension])

  determinant_sigma = tf.math.reduce_prod(sigma_diag, axis=-1)
  determinant_sigma1 = tf.exp(tf.reduce_sum(logvars1, axis=-1))
  determinant_sigma2 = tf.exp(tf.reduce_sum(logvars2, axis=-1))
  term1 = 0.125 * tf.reshape(difference_mus_T @ sigma_mat_inv @ difference_mus, [N, N])
  term2 = 0.5 * tf.math.log(determinant_sigma / tf.sqrt(determinant_sigma1 * determinant_sigma2))
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
