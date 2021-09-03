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

"""Tests for pathdreamer.models.layers."""

from absl.testing import parameterized
from pathdreamer.models import layers
import tensorflow as tf


class LayersTest(tf.test.TestCase, parameterized.TestCase):
  """Tests on the image_models file."""

  @parameterized.parameters((1, 128, 1), (2, 256, 2))
  def test_resstack(self, batch_size, image_size, strides):
    """Tests that ResStack model outputs correct shapes."""
    input_dim = 32
    expansion = 4
    blocks = 2
    output_dim = expansion * input_dim
    test_model = layers.ResStack(input_dim, input_dim, blocks, strides,
                                 expansion)
    test_input = tf.random.uniform(
        (batch_size, image_size, image_size, input_dim), dtype=tf.float32)
    test_output = test_model(test_input)
    self.assertEqual(
        test_output.shape,
        (batch_size, image_size // strides, image_size // strides, output_dim))

  @parameterized.parameters((1, 64, 1), (2, 128, 2))
  def test_resstack_transposed(self, batch_size, image_size, strides):
    """Tests that ResStackTranspose model outputs correct shapes."""
    input_dim = 32
    output_dim = 16
    blocks = 2
    test_model = layers.ResStackTranspose(input_dim, output_dim, blocks,
                                          strides)
    test_input = tf.random.uniform(
        (batch_size, image_size, image_size, input_dim), dtype=tf.float32)
    test_output = test_model(test_input)
    self.assertEqual(
        test_output.shape,
        (batch_size, image_size * strides, image_size * strides, output_dim))

  @parameterized.parameters((4, 8, 8, 3, 2, 64), (4, 8, 8, 3, 2, 32),
                            (4, 16, 8, 1, 1, 32))
  def test_spectral_conv(self, batch_size, input_dims, output_dims, kernel_size,
                         strides, input_size):
    """Tests that spectral convolution outputs are of the correct shapes."""
    # Require TPU / GPU to run grouped convolutions.
    spectral_conv = layers.SpectralConv(
        output_dims, kernel_size=kernel_size, strides=strides)
    # Use a regular conv to determine shape.
    normal_conv = tf.keras.layers.Conv2D(
        output_dims, kernel_size=kernel_size, strides=strides)

    test_input = tf.random.uniform(
        (batch_size, input_size, input_size, input_dims))
    test_output = spectral_conv(test_input)
    normal_output = normal_conv(test_input)
    self.assertAllEqual(test_output.shape, normal_output.shape)

  @parameterized.parameters((1, 3, 2), (4, 5, 1))
  def test_partial_conv(self, batch_size, kernel_size, strides):
    output_dims = 16
    input_dims = 32
    input_size = 32
    partial_conv = layers.PartialConv(
        output_dims, kernel_size=kernel_size, strides=strides)
    normal_conv = tf.keras.layers.Conv2D(
        output_dims, kernel_size=kernel_size, strides=strides)
    test_input = tf.random.uniform(
        (batch_size, input_size, input_size, input_dims))
    test_output, _ = partial_conv(test_input)
    normal_output = normal_conv(test_input)
    self.assertAllEqual(test_output.shape, normal_output.shape)


if __name__ == '__main__':
  tf.test.main()
