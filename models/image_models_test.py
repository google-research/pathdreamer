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

"""Tests for pathdreamer.models.image_models."""

import itertools

from absl.testing import parameterized
from pathdreamer.models import image_models
import tensorflow as tf


class ImageModelsTest(tf.test.TestCase, parameterized.TestCase):
  """Tests on the image_models file."""

  @parameterized.parameters(list(itertools.product((1, 2), (128, 256), (41,))))
  def test_model_output(self, batch_size, image_size, channels):
    """Tests that encoder / decoder outputs correct shapes."""
    test_input = tf.random.uniform(
        (batch_size, image_size, image_size, channels),
        maxval=1,
        dtype=tf.int32)
    test_input = tf.cast(test_input, tf.float32)
    hidden_dims = 8

    test_encoder = image_models.ResNetEncoder(image_size=image_size,
                                              hidden_dims=hidden_dims,
                                              resnet_version='50')
    test_decoder = image_models.ResNetDecoder(
        image_size=image_size,
        hidden_dims=hidden_dims,
        output_dim=channels,
        resnet_version='50')

    test_encoder_output, test_skip = test_encoder(test_input)
    # Encoder output should be a vector of shape (N, output_dim).
    self.assertEqual(test_encoder_output.shape[0], batch_size)
    self.assertLen(test_encoder_output.shape, 2)

    tiled_encoder_output = test_encoder_output[:, None, None, :]
    test_decoder_output = test_decoder(tiled_encoder_output, test_skip)
    # Decoder output should be equal to input shape.
    self.assertEqual(test_decoder_output.shape, test_input.shape)


if __name__ == '__main__':
  tf.test.main()
