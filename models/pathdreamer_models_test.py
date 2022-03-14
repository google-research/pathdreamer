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

"""Tests for pathdreamer.models.pathdreamer_models."""

import itertools

from absl import flags
from absl.testing import parameterized
import numpy as np
from pathdreamer import constants
from pathdreamer.models import pathdreamer_config
from pathdreamer.models import pathdreamer_models
import tensorflow as tf

FLAGS = flags.FLAGS


def diff_proportion(a, b, atol=1e-2):
  diff_count = tf.reduce_sum(tf.cast(tf.abs(a - b) > atol, tf.float32))
  return (diff_count / tf.size(a, out_type=tf.float32)).numpy()


class PathdreamerModelsTest(tf.test.TestCase, parameterized.TestCase):
  """Tests on the pathdreamer_models file."""

  @parameterized.parameters(
      list(itertools.product((1, 2), (128, 256), (False, True), (False, True))))
  def test_pathdreamer_model_output(self, batch_size, image_size,
                                    use_segmentation, output_graph_preds):
    """Tests that encoder / decoder outputs correct shapes."""
    test_rgb = tf.random.uniform((batch_size, image_size, image_size * 2, 3),
                                 maxval=255,
                                 dtype=tf.int32)
    test_rgb = tf.cast(test_rgb, tf.uint8)
    test_segmentation = tf.random.uniform(
        (batch_size, image_size, image_size * 2, 1),
        maxval=constants.NUM_MP3D_CLASSES,
        dtype=tf.int32)
    test_segmentation = tf.cast(test_segmentation, tf.uint8)
    test_depth = tf.random.uniform((batch_size, image_size, image_size * 2),
                                   maxval=1,
                                   dtype=tf.float32)
    test_position = tf.random.normal((batch_size, 3), dtype=tf.float32)
    config = pathdreamer_config.get_test_config()
    config.batch_size = batch_size
    config.image_height = image_size
    config.use_segmentation = use_segmentation
    config.output_graph_preds = output_graph_preds
    model = pathdreamer_models.PathdreamerModel(config)
    model.add_to_memory(test_rgb, test_segmentation, test_depth, test_position,
                        mask_blurred=False)
    # Predict at the original position.
    output_data = model(test_position)
    # Projected should be roughly equal to input (same position).
    rgb_equal = tf.reduce_all(output_data.proj_rgb == test_rgb, axis=-1)
    self.assertGreaterEqual(np.mean(rgb_equal), 0.95)

    self.assertEqual(output_data.proj_semantic.shape,
                     (batch_size, image_size, image_size * 2))
    self.assertEqual(output_data.pred_semantic.shape,
                     (batch_size, image_size, image_size * 2))
    self.assertEqual(output_data.proj_rgb.shape, test_rgb.shape)
    self.assertAllInRange(output_data.proj_rgb, 0, 255)
    self.assertEqual(output_data.pred_rgb.shape, test_rgb.shape)
    self.assertAllInRange(output_data.pred_rgb, 0, 255)
    self.assertEqual(output_data.pred_depth.shape, test_depth.shape)
    self.assertAllInRange(output_data.pred_depth, 0, 1)

  def test_internal_point_cloud_representation(self):
    """Tests Pathdreamer's internal point cloud representation is correct."""
    batch_size = 2
    image_size = 4
    test_rgb = tf.random.uniform((batch_size, image_size, image_size * 2, 3),
                                 maxval=255,
                                 dtype=tf.int32)
    test_rgb = tf.cast(test_rgb, tf.uint8)
    test_segmentation = tf.random.uniform(
        (batch_size, image_size, image_size * 2, 1),
        minval=1,
        maxval=constants.NUM_MP3D_CLASSES,
        dtype=tf.int32)
    test_segmentation = tf.cast(test_segmentation, tf.uint8)

    # Construct a depth image with a plane at 1m along the y-axis.
    offset = 0.5 * np.pi / image_size  # Half a pixel width in radians.
    heading = tf.linspace(-np.pi + offset, np.pi - offset, image_size * 2)
    pitch = tf.linspace(0.5 * np.pi - offset, -0.5 * np.pi + offset,
                        image_size)
    x_depth = (1.0 / tf.math.cos(heading))[None, :]
    depth = x_depth / tf.math.cos(pitch)[:, None]
    depth = tf.where(depth > 0, depth, 0)

    # Construct a second depth image with a plane at 1m along the x-axis.
    depth1 = tf.roll(depth, image_size//2, -1)
    test_depth = tf.concat([depth[None, :], depth1[None, :]], axis=0)
    test_depth /= constants.DEPTH_SCALE

    config = pathdreamer_config.get_test_config()
    config.batch_size = batch_size
    config.image_height = image_size
    config.use_segmentation = False
    config.output_graph_preds = False
    model = pathdreamer_models.PathdreamerModel(config)
    # Move the second position 1m along the x-axis.
    start_position = tf.constant([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                                 dtype=tf.float32)
    model.add_to_memory(
        test_rgb,
        test_segmentation,
        test_depth,
        start_position,
        mask_blurred=False)
    mem = model.get_memory_state()
    pc = mem.rgb_coords
    self.assertEqual(pc.shape, [batch_size, 4, 24])

    # Test the pointcloud for planes parallel to the correct axis.
    for ix, (axis, value) in enumerate([(1, 1), (0, 2)]):
      valid_points = tf.reduce_any(
          mem.rgb[ix] != constants.INVALID_RGB_VALUE, axis=1)
      filtered_pc = tf.gather(pc[ix], tf.where(valid_points), axis=1)[..., 0]
      self.assertAllClose(
          filtered_pc[axis],
          image_size**2 * [value],
          msg='Point cloud values on axis %d should all be %d' % (axis, value))


if __name__ == '__main__':
  tf.test.main()
