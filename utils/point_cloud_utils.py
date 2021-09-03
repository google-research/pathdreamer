# Copyright 2021 Google LLC.
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

"""Utility functions for processing 3D point clouds."""

from typing import Tuple
import numpy as np
from pathdreamer import constants
import tensorflow as tf


def get_intrinsic_matrix(hfov: float) -> tf.Tensor:
  """Returns the intrinsic for a given horizontal FOV."""
  return tf.constant([
      [1 / np.tan(hfov / 2.), 0., 0., 0.],
      [0., 1 / np.tan(hfov / 2.), 0., 0.],
      [0., 0., 1, 0],
      [0., 0., 0, 1]], dtype=tf.float32)


def get_filtered_coords_and_feats(feats: tf.Tensor, depth: tf.Tensor,
                                  depth_scale: float):
  """Filter and return valid coordinates and features.

  Coordinates/features that are not visible due to invalid depth are given a
  feature value of zero (which is assumed to be the void class).

  Args:
    feats: (N, H, W) or (N, H, W, C) tensor of feature values.
    depth: (N, H, W) tensor of depth values, with values in [0, 1].
    depth_scale: Maximum depth in metres.

  Returns:
    xyz: (N, 4, H * W) tensor of (x, y, z, 1) coordinate values.
    filtered_feats: (N, H * W) tensor or (N, H * W, C) tensor of filtered
      features. Output shape is dependent on the input shape.
  """
  if len(feats.shape) != 3 and len(feats.shape) != 4:
    raise ValueError('feats should have shape (N, H, W) or (N, H, W, C),'
                     f' got {feats.shape} instead.')
  is_scalar_feat = len(feats.shape) == 3
  if is_scalar_feat:
    feats = feats[..., None]  # Unsqueeze last dimension to act as a channel.
  batch_size, height, width = depth.shape
  channels = feats.shape[-1]
  # Create an approximation for the true world coordinates.
  # [-1, 1] for x and [1, -1] for y as array is y-down while world is y-up.
  xs, ys = tf.meshgrid(tf.linspace(-1, 1, width), tf.linspace(-1, 1, height))
  xs = tf.reshape(tf.cast(xs, tf.float32), (1, 1, height, width))
  xs = tf.tile(xs, [batch_size, 1, 1, 1])
  ys = tf.reshape(tf.cast(ys, tf.float32), (1, 1, height, width))
  ys = tf.tile(ys, [batch_size, 1, 1, 1])
  depth = (depth * depth_scale)[:, None, :, :]
  ones = tf.ones_like(depth)
  # Create xyz coord tensor.
  xyz = tf.concat([xs * depth, ys * depth, depth, ones], axis=1)

  # Find coordinates with valid depth values.
  depth = tf.reshape(depth, (batch_size, -1))
  depth_mask = tf.math.logical_and(depth > 0, depth < depth_scale)

  # Set features to zero for invalid depth values.
  filtered_feats = tf.reshape(feats, (batch_size, -1, channels))
  filtered_feats = filtered_feats * tf.cast(depth_mask[..., None], tf.int32)
  filtered_feats = tf.cast(filtered_feats, tf.float32)

  # Project coordinates into camera view.
  intrinsic_matrix = get_intrinsic_matrix(constants.HFOV)
  xyz = tf.reshape(xyz, (batch_size, 4, -1))
  xyz = xyz * tf.cast(depth_mask[:, None, :], tf.float32)
  xyz = tf.matmul(tf.linalg.inv(intrinsic_matrix), xyz)

  # Remove channels dimension if initial feature was scalar.
  if is_scalar_feat:
    filtered_feats = filtered_feats[..., 0]
  return xyz, filtered_feats


def project_to_feat(
    transformed_coords: tf.Tensor,
    feats: tf.Tensor,
    height: int,
    width: int,
    depth_scale: float,
    input_void_class: float,
    output_void_class: float = 0,
) -> Tuple[tf.Tensor, tf.Tensor]:
  """Project a set of features to a transformed coordinate space.

  Args:
    transformed_coords: Tensor of shape (N, 4, M) of (x, y, z, 1) values, where
      M denotes the number of data points in the point cloud.
    feats: Tensor of shape (N, M) or (N, M, C) of feature values corresponding
      to each location.
    height: Image height in pixels.
    width: Image width in pixels.
    depth_scale: Maximum depth in meters. Values above this are clipped.
    input_void_class: Feature value (class label) that represents an invalid
      point in the input feats.
    output_void_class: Feature value to use in output projected_feat for an
      invalid pixel. By default 0 is used since this corresponds to the void
      class for Matterport segmentations and black pixels in an RGB image.

  Returns:
    projected_depth: Tensor of shape (N, H, W) of depth values in [0, 1].
    projected_feat: Tensor of shape (N, H, W) or (N, H, W, C) of projected
      feature values. Output shape is dependent on the input shape.
  """
  if len(feats.shape) != 2 and len(feats.shape) != 3:
    raise ValueError('feats should have shape (N, M) or (N, M, C), got'
                     f' {feats.shape} instead.')
  is_scalar_feat = len(feats.shape) == 2
  if is_scalar_feat:
    feats = feats[..., None]  # Unsqueeze last dimension to act as a channel.
  channels = feats.shape[-1]
  batch_size = transformed_coords.shape[0]
  # Normalize x, y values by depth.
  depth = transformed_coords[:, 2, :]
  view_coords = tf.math.divide_no_nan(
      transformed_coords[:, 0:2, :], depth[:, None, ...])
  dtype = transformed_coords.dtype

  # Find all valid coordinates.
  denorm_coords = tf.cast(
      tf.stack([(view_coords[:, 0, :] + 1) / 2 * tf.cast(width, dtype),
                (view_coords[:, 1, :] + 1) / 2 * tf.cast(height, dtype)],
               axis=1), tf.int32)
  valid_coords = tf.math.logical_and(
      tf.math.logical_and(denorm_coords[:, 0, :] >= 0,
                          denorm_coords[:, 0, :] < width),
      tf.math.logical_and(denorm_coords[:, 1, :] >= 0,
                          denorm_coords[:, 1, :] < height))
  # Exclude points that are behind the camera or have no depth return.
  valid_coords = tf.math.logical_and(valid_coords, depth > 0)
  # Exclude points that are void class.
  valid_feats = tf.reduce_all(feats != input_void_class, axis=-1)
  valid_coords = tf.math.logical_and(valid_coords, valid_feats)
  # Convert to a 1D tensor for scattering.
  batch_offset = tf.range(0, batch_size)[:, None] * width * height
  flat_coords = (batch_offset + denorm_coords[:, 1, :] * width +
                 denorm_coords[:, 0, :]) * tf.cast(valid_coords, tf.int32)
  flat_coords = tf.reshape(flat_coords, (-1,))
  flat_depth = tf.reshape(depth, (-1,))

  # Calculate reprojected depth image
  scattered_depth = tf.tensor_scatter_nd_min(
      tf.cast(tf.fill((batch_size * height * width, 1), depth_scale), dtype),
      flat_coords[:, None], flat_depth[..., None])
  projected_depth = tf.reshape(scattered_depth, (batch_size, height, width))
  projected_depth = tf.clip_by_value(
      projected_depth, 0, depth_scale) / depth_scale

  # A lot of points in the cloud collide when mapped to pixel space.
  # Gather from the depth map to identify which points had the minimum depth
  # in pixel space, discard the others to avoid collisions when reprojecting
  # segmentation classes.
  min_depth = tf.gather(scattered_depth, flat_coords)[..., 0]
  flat_coords = flat_coords * tf.cast(flat_depth < min_depth + 0.1, tf.int32)

  # Calculate reprojected feature image
  flat_feats = tf.reshape(feats, (-1, channels))
  scattered_feat = tf.tensor_scatter_nd_max(
      tf.cast(
          tf.fill((batch_size * height * width, channels), output_void_class),
          dtype), flat_coords[:, None], flat_feats)
  projected_feat = tf.reshape(scattered_feat,
                              (batch_size, height, width, channels))

  # Remove channels dimension if initial feature was scalar.
  if is_scalar_feat:
    projected_feat = projected_feat[..., 0]
  return projected_depth, projected_feat

