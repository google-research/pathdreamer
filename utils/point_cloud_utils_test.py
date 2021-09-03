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

"""Tests for pathdreamer.utils.point_cloud_utils."""

from absl.testing import parameterized
from pathdreamer import constants
from pathdreamer.utils import point_cloud_utils
import tensorflow as tf


class PointCloudUtilsTest(tf.test.TestCase, parameterized.TestCase):
  """Tests on the point_cloud_utils file."""

  @parameterized.parameters((2, 64,), (1, 128,))
  def test_filtered_coords_and_feats(self, batch_size, image_size):
    """Tests that valid results are returned for the filtering function."""
    test_feats = tf.random.uniform((batch_size, image_size, image_size),
                                   0, constants.NUM_MP3D_CLASSES,
                                   dtype=tf.int32)
    test_depth = tf.random.uniform((batch_size, image_size, image_size),
                                   0, constants.DEPTH_SCALE,
                                   dtype=tf.float32)
    xyz1, filtered_feats = point_cloud_utils.get_filtered_coords_and_feats(
        test_feats, test_depth, constants.DEPTH_SCALE)
    self.assertEqual(xyz1.shape, (batch_size, 4, image_size * image_size))
    self.assertEqual(
        filtered_feats.shape, (batch_size, image_size * image_size))
    self.assertAllInRange(filtered_feats, 0, constants.NUM_MP3D_CLASSES)

  @parameterized.parameters((2, 64, False), (1, 128, False), (2, 64, True),
                            (1, 128, True))
  def test_project_to_feat(self, batch_size, image_size, multi_channel):
    """Tests that valid results are returned for the projection function."""
    feat_shape = (batch_size, image_size, image_size)
    if multi_channel:
      feat_shape = (batch_size, image_size, image_size, 3)
    test_feats = tf.random.uniform(
        feat_shape, 0, constants.NUM_MP3D_CLASSES, dtype=tf.int32)
    test_depth = tf.random.uniform((batch_size, image_size, image_size),
                                   0, constants.DEPTH_SCALE,
                                   dtype=tf.float32)
    xyz1, filtered_feats = point_cloud_utils.get_filtered_coords_and_feats(
        test_feats, test_depth, constants.DEPTH_SCALE)
    projected_depth, projected_feat = point_cloud_utils.project_to_feat(
        xyz1, filtered_feats, image_size, image_size,
        constants.DEPTH_SCALE, constants.INVALID_SEM_VALUE)
    self.assertEqual(projected_depth.shape,
                     (batch_size, image_size, image_size))
    self.assertAllInRange(projected_depth, 0, 1)
    self.assertEqual(projected_feat.shape, feat_shape)
    self.assertAllInRange(projected_feat, tf.reduce_min(test_feats),
                          tf.reduce_max(test_feats))


if __name__ == '__main__':
  tf.test.main()
