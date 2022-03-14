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

"""Utils function for panorama processing."""

import math
from typing import Optional, Tuple

import numpy as np
from pathdreamer.utils import point_cloud_utils
import tensorflow as tf
from tensorflow_addons import image as tfa_image


def get_world_to_image_transform(image_shape,
                                 fov,
                                 camera_intrinsics: Optional[tf.Tensor] = None,
                                 rotations: Optional[Tuple[int, int]] = None,
                                 rotation_matrix: Optional[tf.Tensor] = None):
  """Returns a 3x3 transformation matrix from world to image coordinates.

  The image is oriented orthogonally to the x axis. The x axis of the image
  points away from the z axis in world coordinates, and the y axis of the image
  points away from the y axis in world coordinates.

  Modified from //experimental/earthsea/wanderer/geometry_utils.py.

  Args:
    image_shape: list with shape of the image (height, width).
    fov: tensor with the fields of view of the image (vertical, horizontal) in
      radians.
    camera_intrinsics: Optional camera intrinsics matrix to use instead of
      computing it using field of view.
    rotations: optional tensor containing pitch and heading in radians for
      rotating camera. A positive pitch rotates the camera upwards. A positive
      heading rotates the camera along the equator clockwise.
    rotation_matrix: Optional 3x3 rotation matrix to use instead as an
      alternative to the rotations parameter.

  Returns:
    A 3x3 tensor that transforms world to image coordinates.
  """
  if camera_intrinsics is None:
    height, width = image_shape
    fov_y, fov_x = tf.unstack(fov)

    fx = 0.5 * (width - 1.0) / tf.tan(fov_x / 2)
    fy = 0.5 * (height - 1.0) / tf.tan(fov_y / 2)

    camera_intrinsics = tf.stack([
        tf.stack([fx, 0, 0.5 * (width - 1)]),
        tf.stack([0, fy, 0.5 * (height - 1)]),
        tf.stack([0., 0, 1])
    ])
  if rotations is not None:
    rot_pitch, rot_heading = tf.unstack(rotations)
    pitch_rotation = tf.stack([
        tf.stack([1., 0, 0]),
        tf.stack([0, tf.cos(-rot_pitch), -tf.sin(-rot_pitch)]),
        tf.stack([0, tf.sin(-rot_pitch), tf.cos(-rot_pitch)])
    ])
    heading_rotation = tf.stack([
        tf.stack([tf.cos(-rot_heading), 0, tf.sin(-rot_heading)]),
        tf.stack([0., 1, 0]),
        tf.stack([-tf.sin(-rot_heading), 0, tf.cos(-rot_heading)])
    ])
    extrinsics = pitch_rotation @ heading_rotation
  elif rotation_matrix is not None:
    extrinsics = rotation_matrix
  else:
    extrinsics = tf.constant([
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
    ])

  transform = camera_intrinsics @ extrinsics
  return transform


def equirectangular_pixel_rays(output_height):
  """Generates a 3d point on a unit ball for each equirectangular image pixel.

  The output coordinate system is x-right, y-down, z-forward at the center of
  the equirectangular image.

  Args:
    output_height: Int height of the equirectangular image.

  Returns:
    pixel_rays: an [3, output_height * output_width] tensor containing
      an xyz coordinate on the unit-radius ball for each pixel.
  """

  output_width = tf.cast(tf.cast(output_height, tf.float32) * 2, tf.int32)
  heading = tf.linspace(-math.pi, math.pi, output_width)
  pitch = tf.linspace(0.0, math.pi, output_height)
  heading, pitch = tf.meshgrid(heading, pitch)
  xs = tf.sin(pitch) * tf.sin(heading)
  ys = -tf.cos(pitch)
  zs = tf.sin(pitch) * tf.cos(heading)
  pixel_rays = tf.reshape(tf.stack([xs, ys, zs], axis=0), ((3, -1)))
  return pixel_rays


def project_feats_to_equirectangular(
    feats: tf.Tensor, xyz1: tf.Tensor, height: int, width: int,
    void_class: float,
    depth_scale: float,
) -> Tuple[tf.Tensor, tf.Tensor]:
  """Project point cloud feats / coords into an equirectangular image.

  Args:
    feats: (N, M) or (N, M, C) tensor of semantic segmentation features.
    xyz1: (N, 4, M): xyz1 coordinates.
    height: Height in pixels of the projected image.
    width: Width in pixels of the projected image.
    void_class: Feature value (class label) that represents an empty pixel.
    depth_scale: Maximum depth in meters. Values above this are clipped.

  Returns:
    reprojected_depth: (N, H, W) tensor of equirectangular image of depth
      values in [0, 1].
    reprojected_feats: (N, H, W) tensor of equirectangular image of features.
  """
  # Map frame coords to the new location.
  # relative_pos indicates the relative displacement of frame i wrt frame 0.
  x, y, z = xyz1[:, 0, :], xyz1[:, 1, :], xyz1[:, 2, :]
  rad = (x**2 + y**2 + z**2)**0.5
  # Heading as defined from the x-axis, which is between the center of the pano
  # image and the right hand edge and rotating left.
  heading = tf.math.atan2(y, x)
  # Heading redefined from the left hand edge of the image and rotating right.
  heading = 1.5 * math.pi - heading
  dtype = xyz1.dtype
  # Map to [0, 2pi] domain.
  heading = heading + (2 * math.pi) * tf.cast(heading <= 0, dtype)
  heading = heading - (2 * math.pi) * tf.cast(heading > (2 * math.pi), dtype)
  elevation = tf.math.acos(tf.math.divide_no_nan(z, rad))

  # Map to 360 panorama image coordinates.
  proj_x = rad * ((heading / (2 * math.pi)) * 2 - 1)
  proj_y = rad * ((elevation / math.pi) * 2 - 1)
  proj_z = rad
  proj_xyz1 = tf.stack([proj_x, proj_y, proj_z, tf.ones_like(proj_x)], axis=1)

  reprojected_depth, reprojected_feats = point_cloud_utils.project_to_feat(
      tf.cast(proj_xyz1, dtype), tf.cast(feats, dtype), height, width,
      depth_scale=depth_scale, input_void_class=void_class)
  return reprojected_depth, reprojected_feats


def equirectangular_to_pointcloud(
    feats: tf.Tensor,
    depth: tf.Tensor,
    void_class: float,
    depth_scale: float,
    size_mult: float = 1.0,
    interpolation_method: str = 'nearest') -> Tuple[tf.Tensor, tf.Tensor]:
  """Filter and return valid coords and features for equirectangular image.

  Coordinates/features that are not visible due to invalid depth are still
  returned but given a feature value of void_class and an xyz1 coordinate of
  (0, 0, 0, 1).

  Args:
    feats: (N, H, W) or (N, H, W, C) tensor of feature values.
    depth: (N, H, W) tensor of depth values, with values in [0, 1].
    void_class: feature value to use for invalid points in the output.
    depth_scale: Maximum depth in metres.
    size_mult: Amount of upscale the features / depths by. This creates denser
      point clouds.
    interpolation_method: Interpolation method for resizing features when
      size_mult != 1.0.
  Returns:
    xyz1: (N, 4, H * W) tensor of (x, y, z, 1) coordinate values.
    filtered_feats: (N, H * W) or (N, H * W, C) tensor of filtered features.
  """
  if len(feats.shape) != 3 and len(feats.shape) != 4:
    raise ValueError('feats should have shape (N, H, W) or (N, H, W, C),'
                     f' got {feats.shape} instead.')
  if void_class < 0.0 and feats.dtype in [
      tf.uint8, tf.uint16, tf.uint32, tf.uint64
  ]:
    raise ValueError(
        'feats datatype must be signed if the void class is negative')
  is_scalar_feat = len(feats.shape) == 3
  if is_scalar_feat:
    feats = feats[..., None]  # Unsqueeze last dimension to act as a channel.
  batch_size, height, width, channels = feats.shape
  assert width == 2 * height, 'Expected equirectangular input images'
  scaled_height = int(height * size_mult)
  scaled_width = int(width * size_mult)
  pano_depth = tf.image.resize(
      depth[..., None], (scaled_height, scaled_width), method='nearest')[..., 0]
  pano_feats = tf.image.resize(
      feats, (scaled_height, scaled_width), method=interpolation_method)
  dtype = depth.dtype
  # Add points to point cloud memory.
  half_pixel_width = 0.5 * np.pi / scaled_height
  elevation = tf.cast(
      tf.linspace(half_pixel_width, np.pi - half_pixel_width, scaled_height),
      dtype)
  # Define heading from the x-axis, increasing towards the y-axis.
  heading = tf.cast(
      tf.linspace(1.5 * np.pi - half_pixel_width,
                  -0.5 * np.pi + half_pixel_width, scaled_width), dtype)
  # Mask out invalid depths.
  depth_mask = tf.cast(tf.math.logical_and(pano_depth > 0, pano_depth < 1.0),
                       dtype)
  rad = (pano_depth * depth_scale) * depth_mask
  pano_feats = tf.where(depth_mask[..., None] == 0, void_class, pano_feats)

  # Move to correct relative position.
  x = rad * tf.math.sin(elevation)[:, None] * tf.math.cos(heading)[None, :]
  y = rad * tf.math.sin(elevation)[:, None] * tf.math.sin(heading)[None, :]
  z = rad * tf.math.cos(elevation)[:, None]
  xyz1 = tf.stack([
      tf.reshape(x, (batch_size, -1)),
      tf.reshape(y, (batch_size, -1)),
      tf.reshape(z, (batch_size, -1)),
      tf.ones(
          (batch_size, scaled_height * scaled_width),
          dtype=dtype)
  ], axis=1)
  filtered_feats = tf.reshape(pano_feats, (batch_size, -1, channels))

  # Remove channels dimension if initial feature was scalar.
  if is_scalar_feat:
    filtered_feats = filtered_feats[..., 0]
  return xyz1, filtered_feats


def mask_pano(pano: tf.Tensor,
              proportion: float = 0.125,
              masked_region_value=0):
  """Applies masking to the top and bottom regions of a panorama.

  Args:
    pano: (N, H, W, C) tensor to be masked.
    proportion: Proportion of the rows of an image that should be masked from
      the top and bottom of the image. The total proportion masked will be
      (proportion * 2).
    masked_region_value: Value to set for masked regions.
  Returns:
    masked: (N, H, W, C) tensor.
  """
  _, height, _, _ = pano.shape
  masked_height = int(height  * proportion)
  height_range = tf.range(0, height)
  mask = tf.math.logical_and(height_range >= masked_height,
                             height_range <= height - masked_height)
  mask = tf.cast(mask, pano.dtype)[None, :, None, None]
  return mask * pano + (1 - mask) * masked_region_value  # (N, H, W, C)


def crop_pano(pano: tf.Tensor,
              proportion: float = 0.125,
              method: str = tf.image.ResizeMethod.BILINEAR) -> tf.Tensor:
  """Applies cropping to remove the top and bottom `proportion` of the image.

  The result is resized to maintain the original image dimensions.

  Args:
    pano: (N, H, W, C) tensor to be masked.
    proportion: Proportion of the rows of an image that should be masked from
      the top and bottom of the image. The total proportion masked will be
      (proportion * 2).
    method: Interpolation method to use for resizing.

  Returns:
    masked: (N, H, W, C) tensor.
  """
  if len(pano.shape) == 3:
    height, width, _ = pano.shape
  elif len(pano.shape) == 4:
    _, height, width, _ = pano.shape
  else:
    raise ValueError(
        f'pano should be of shape (N, H, W, C), got {pano.shape} instead.')
  masked_height = int(height * proportion)
  cropped = tf.image.crop_to_bounding_box(pano, masked_height, 0,
                                          height - masked_height, width)
  cropped = tf.image.resize(
      cropped, (height, width), method=method, antialias=True)
  cropped = tf.cast(cropped, pano.dtype)
  return cropped


def rotate_pano(pano: tf.Tensor,
                matrix: tf.Tensor,
                output_height: Optional[int] = None) -> tf.Tensor:
  """Rotates an equirectangular panorama using a provided 3x3 rotation matrix.

  Args:
    pano: (N, H, W, C) tensor of equirectangular images to be rotated.
    matrix: (N, 3, 3) tensor of 3x3 rotation matrices.
    output_height: Optional integer height of output pano, used for resizing.

  Returns:
    rotated_pano: (N, H, W, C) tensor.
  """
  output_shape = pano.shape
  if output_shape[2] != output_shape[1] * 2:
    raise ValueError('Pano width must be twice height.')
  if output_height is not None:
    output_shape[1] = output_height
    output_shape[2] = int(2 * output_height)

  pixel_rays = equirectangular_pixel_rays(output_shape[1])
  rotated_pixel_rays = tf.matmul(matrix, tf.expand_dims(pixel_rays, 0))

  x = rotated_pixel_rays[:, 0]
  y = rotated_pixel_rays[:, 1]
  z = rotated_pixel_rays[:, 2]
  pitch = tf.acos(-y)
  heading = tf.atan2(x, z)

  heading_pixels = (heading / (2 * math.pi) + 0.5) * (pano.shape[2] - 1)
  pitch_pixels = pitch / math.pi * (pano.shape[1] - 1)

  image_coordinates = tf.stack([pitch_pixels, heading_pixels], axis=-1)
  rotated_pano = tfa_image.interpolate_bilinear(pano, image_coordinates)
  rotated_pano = tf.reshape(rotated_pano, output_shape)
  return rotated_pano
