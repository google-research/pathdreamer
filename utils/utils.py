# Copyright 2022 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions."""

from typing import Optional, Tuple
import numpy as np
import tensorflow as tf


def create_label_colormap() -> np.ndarray:
  """Creates a label colormap supporting up to 256 labels.

  Returns:
    A colormap for visualizing segmentation results. Colors are expected to be
      distinct, and aid in the visualization of segmentation maps.
  """
  colormap = np.zeros((256, 3), dtype=int)
  ind = np.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((ind >> channel) & 1) << shift
    ind >>= 3

  return colormap


def cmap_to_label(image_tensor, cmap):
  """Maps an image to labels.

  This is the inverse of create_label_colormap().

  Args:
    image_tensor: (H, W, 3) tensor or array of RGB image.
    cmap: (256, 3) array of color map.

  Returns:
    image_labels: (H, W) int tensor of semantic labels.
  """
  rgb_equal = np.all(image_tensor[..., None, :] == cmap, axis=-1)
  image_labels = np.argmax(rgb_equal, axis=-1)
  return image_labels


def compute_kl(mu1: tf.Tensor, logvar1: tf.Tensor, mu2: tf.Tensor,
               logvar2: tf.Tensor) -> tf.Tensor:
  """Computes Kullback-Leibler Divergence for two distributions P and Q.

  Args:
    mu1: Tensor indicating mean value of P. Multi-dimensional tensors are valid.
    logvar1: Tensor indicating log(variance) of P. Expected to have the same
      shape as mu1.
    mu2: Tensor indicating mean value of Q. Expected to have the same shape as
      mu1.
    logvar2: Tensor indicating log(variance) of Q. Expected to have the same
      shape as mu1.

  Raises:
    ValueError: If the argument shapes are not equal.

  Returns:
    kld: Tensor of same shape as mu1 indicating KL divergence between P and Q.
  """
  if not mu1.shape == logvar1.shape == mu2.shape == logvar2.shape:
    raise ValueError('Arguments to compute KLD should have the same shape.')

  sigma1 = tf.math.exp(0.5 * logvar1)
  sigma2 = tf.math.exp(0.5 * logvar2)
  return (tf.math.log(sigma2 / sigma1) + (tf.math.exp(logvar1) +
                                          (mu1 - mu2)**2) /
          (2 * tf.math.exp(logvar2)) - 1 / 2)


def reparameterize(mu: tf.Tensor, logvar: tf.Tensor) -> tf.Tensor:
  """Reparameterization trick to draw random vector from N(mu, sigma)."""
  sigma = tf.math.exp(0.5 * logvar)
  eps = tf.random.normal(sigma.shape, dtype=mu.dtype)
  return eps * sigma + mu


def compute_sequence_iou(
    one_hot_pred: tf.Tensor,
    one_hot_true: tf.Tensor,
    mask: tf.Tensor,
    spatial_mask: Optional[tf.Tensor] = None) -> Tuple[tf.Tensor, tf.Tensor]:
  """Computes mean intersection-over-union between two one hot sequences.

  Args:
    one_hot_pred: One-hot encoded tensor of shape (N, T, H, W, C).
    one_hot_true: One-hot encoded tensor of shape (N, T, H, W, C).
    mask: Tensor of shape (N, T) with values 0 if an item is padding (and 1
      otherwise).
    spatial_mask: Tensor of shape (N, T, H, W) with values 0 if an item is
      masked out and 1 otherwise.

  Returns:
    seq_iou: Tensor of shape (N, T) representing mIOU of each frame.
    mean_iou: mIOU of the entire sequence (taking mask into account).
  """
  # If no mask provided, use all values.
  if spatial_mask is None:
    spatial_mask = tf.ones_like(one_hot_pred)[..., 0]
  intersect = tf.reduce_sum(
      one_hot_pred * one_hot_true * spatial_mask[..., None], axis=(2, 3, 4))
  union = tf.reduce_sum(
      (one_hot_pred + one_hot_true) * spatial_mask[..., None],
      axis=(2, 3, 4)) - intersect
  seq_iou = tf.math.divide_no_nan(intersect * mask, union * mask)
  mask_length = tf.reduce_sum(mask, axis=1)
  mean_iou = tf.math.divide_no_nan(tf.reduce_sum(seq_iou, axis=1), mask_length)
  return seq_iou, tf.reduce_mean(mean_iou)


def compute_sequence_accuracy(
    class_pred: tf.Tensor,
    class_gt: tf.Tensor,
    mask: tf.Tensor,
    spatial_mask: Optional[tf.Tensor] = None) -> Tuple[tf.Tensor, tf.Tensor]:
  """Computes mean intersection-over-union between two one hot sequences.

  Args:
    class_pred: Prediction tensor of shape (N, T, H, W).
    class_gt: Label tensor of shape (N, T, H, W).
    mask: Tensor of shape (N, T) with values 0 if an item is padding (and 1
      otherwise).
    spatial_mask: Tensor of shape (N, T, H, W) with values 0 if an item is
      masked out and 1 otherwise.

  Returns:
    seq_accuracy: Tensor of shape (N, T) representing accuracy of each frame.
    mean_accuracy: accuracy of the entire sequence (taking mask into account).
  """
  # If no mask provided, use all values.
  if spatial_mask is None:
    spatial_mask = tf.ones_like(class_pred)
  equal = tf.cast(class_pred == class_gt, spatial_mask.dtype) * spatial_mask
  seq_accuracy = tf.math.divide_no_nan(
      tf.cast(tf.reduce_sum(equal, axis=(2, 3)), tf.float32),
      tf.cast(tf.reduce_sum(spatial_mask, axis=(2, 3)), tf.float32))
  mask_length = tf.reduce_sum(mask, axis=1)
  mean_accuracy = tf.math.divide_no_nan(
      tf.reduce_sum(seq_accuracy, axis=1), mask_length)
  return seq_accuracy, tf.reduce_mean(mean_accuracy)


def nearest_neighbor_inpaint(image, void_class: int = 0):
  """Fills zero pixels in an image with the closest non-zero value.

  Args:
    image: (N, H, W) tensor of a batch of 2D pixel values.
    void_class: Value in image to inpaint.
  Returns:
    filled: (N, H, W) tensor of inpainted image.
  """
  def inpaint_single_image(image):
    nonzero_coords = tf.where(image != void_class)
    zero_coords = tf.where(image == void_class)
    diff = nonzero_coords[:, None, :] - zero_coords[None, ...]
    distance = tf.reduce_sum(diff ** 2, axis=-1)
    closest_nonzero = tf.argmin(distance, axis=0)
    closest_coords = tf.gather_nd(nonzero_coords, closest_nonzero[..., None])
    nonzero_values = tf.gather_nd(image[..., None], closest_coords)
    filled = tf.tensor_scatter_nd_update(image[..., None], zero_coords,
                                         nonzero_values)[..., 0]
    return filled

  return tf.map_fn(inpaint_single_image, image, parallel_iterations=4)

