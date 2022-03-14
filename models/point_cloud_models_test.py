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

"""Tests for pathdreamer.models.point_cloud_models."""

import itertools

from absl.testing import parameterized
from pathdreamer import constants
from pathdreamer.models import point_cloud_models
import tensorflow as tf


class PointCloudModelsTest(tf.test.TestCase, parameterized.TestCase):
  """Tests on the point_cloud_models file."""

  @parameterized.parameters(
      list(itertools.product(
          (1, 2), (3,), (128, 256), (True, False), (True, False),
          ('50', '101', '152'))))
  def test_inpainter_output(self, batch_size, seq_len, image_height,
                            circular_pad, random_noise, resnet_version):
    """Tests that encoder / decoder outputs correct shapes."""
    test_classes = tf.random.uniform(
        (batch_size, seq_len, image_height, image_height),
        maxval=constants.NUM_MP3D_CLASSES,
        dtype=tf.int32)
    test_rgb = tf.one_hot(test_classes, constants.NUM_MP3D_CLASSES)
    test_depth = tf.random.uniform(
        (batch_size, seq_len, image_height, image_height, 1),
        maxval=10.0,
        dtype=tf.float32)

    inpainter = point_cloud_models.ResNetSeqInpainter(
        hidden_dims=8, resnet_version=resnet_version, image_height=image_height,
        circular_pad=circular_pad, random_noise=random_noise)
    output_data = inpainter([test_rgb, test_depth])
    # Encoder output should be a vector of shape (N, output_dim).
    self.assertEqual(output_data.pred_logits.shape, test_rgb.shape)
    self.assertEqual(output_data.pred_depth.shape, test_depth.shape)


if __name__ == '__main__':
  tf.test.main()
