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

"""Tests for pathdreamer.utils.pano_utils."""

from absl import flags
from absl.testing import parameterized
from pathdreamer import constants
from pathdreamer.utils import pano_utils
import tensorflow as tf


FLAGS = flags.FLAGS


def diff_proportion(a, b, atol=1e-2):
  diff_count = tf.reduce_sum(tf.cast(tf.abs(a - b) > atol, tf.float32))
  return (diff_count / tf.size(a, out_type=tf.float32)).numpy()


class PanoUtilsTest(tf.test.TestCase, parameterized.TestCase):
  """Tests on the pano_utils file."""

  def test_equirectangular_pixel_rays(self):
    """Tests the coordinate system returned by equirectangular_pixel_rays."""
    pixel_rays = pano_utils.equirectangular_pixel_rays(3)
    pixel_rays = tf.reshape(tf.transpose(pixel_rays), (3, 6, 3))
    expected_pixel_rays = tf.constant([
        [
            [0.0, -1.0, 0.0],  # All pointing up.
            [0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0]
        ],
        [
            [0.0, 0.0, -1.0],  # Sweeping the horizon from left to right.
            [-9.5105648e-01, 4.3711388e-08, -3.0901703e-01],
            [-5.8778524e-01, 4.3711388e-08, 8.0901694e-01],
            [5.8778524e-01, 4.3711388e-08, 8.0901694e-01],
            [9.5105648e-01, 4.3711388e-08, -3.0901703e-01],
            [0.0, 0.0, -1.0]
        ],
        [
            [0.0, 1.0, 0.0],  # All pointing down.
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0]
        ]
    ])
    self.assertAllClose(pixel_rays, expected_pixel_rays)

  @parameterized.parameters((2, 64), (1, 128))
  def test_feats_to_equirectangular(self, batch_size, image_size):
    """Tests that valid results are returned for the projection function."""
    num_points = image_size**2
    feats = tf.random.uniform((batch_size, num_points),
                              0,
                              constants.NUM_MP3D_CLASSES,
                              dtype=tf.int32)
    xyz = tf.random.normal((batch_size, 3, num_points))
    xyz1 = tf.concat([xyz, tf.ones((batch_size, 1, num_points))], axis=1)
    reprojected_depth, reprojected_feats = (
        pano_utils.project_feats_to_equirectangular(
            feats, xyz1, image_size, image_size * 2,
            constants.INVALID_SEM_VALUE, constants.DEPTH_SCALE))
    self.assertAllEqual(reprojected_depth.shape,
                        (batch_size, image_size, image_size * 2))
    self.assertAllInRange(reprojected_depth, 0, 1)
    self.assertAllEqual(reprojected_feats.shape,
                        (batch_size, image_size, image_size * 2))
    self.assertAllInRange(reprojected_feats, 0,
                          constants.NUM_MP3D_CLASSES)

  @parameterized.parameters((2, 64, False), (1, 128, False), (2, 64, True),
                            (1, 128, True))
  def test_filter_equirectangular(self, batch_size, image_size, multi_channel):
    """Tests that valid results are returned for the projection function."""
    feat_shape = (batch_size, image_size, 2 * image_size)
    channels_dim = 3
    if multi_channel:
      feat_shape = (batch_size, image_size, 2 * image_size, channels_dim)
    feats = tf.random.uniform(
        feat_shape, 0, constants.NUM_MP3D_CLASSES, dtype=tf.int32)
    depth = tf.random.uniform((batch_size, image_size, 2 * image_size),
                              0,
                              constants.DEPTH_SCALE,
                              dtype=tf.float32)
    xyz1, filtered_feats = pano_utils.equirectangular_to_pointcloud(
        feats, depth, constants.INVALID_SEM_VALUE, constants.DEPTH_SCALE)
    self.assertAllEqual(xyz1.shape, (batch_size, 4, 2 * image_size**2))
    if multi_channel:
      self.assertAllEqual(filtered_feats.shape,
                          (batch_size, 2 * image_size**2, channels_dim))
    else:
      self.assertAllEqual(filtered_feats.shape, (batch_size, 2 * image_size**2))
    self.assertAllInRange(filtered_feats, 0, constants.NUM_MP3D_CLASSES)

  @parameterized.parameters((2, 64, tf.float32), (2, 64, tf.int32),
                            (1, 256, tf.int32))
  def test_mask_pano(self, batch_size, image_height, dtype):
    """Tests that valid results are returned for pano masking."""
    pano = tf.random.uniform(
        (batch_size, image_height, image_height * 2, 3), 0, 255, dtype=dtype)
    masked_pano = pano_utils.mask_pano(pano)

    self.assertAllEqual(masked_pano.shape, pano.shape)
    self.assertEqual(masked_pano.dtype, pano.dtype)
    # Check top and bottom row of image to make sure it's masked.
    self.assertAllInSet(masked_pano[:, 0, ...], [0])
    self.assertAllInSet(masked_pano[:, -1, ...], [0])

  @parameterized.parameters((2, 64, tf.float32), (2, 64, tf.int32),
                            (1, 256, tf.int32))
  def test_crop_pano(self, batch_size, image_height, dtype):
    """Tests that valid results are returned for pano cropping."""
    pano = tf.random.uniform(
        (batch_size, image_height, image_height * 2, 3), 0, 255, dtype=dtype)
    cropped_pano = pano_utils.crop_pano(pano)
    self.assertAllEqual(cropped_pano.shape, pano.shape)
    self.assertEqual(cropped_pano.dtype, pano.dtype)


if __name__ == '__main__':
  tf.test.main()
