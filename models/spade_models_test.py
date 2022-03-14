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

"""Tests for pathdreamer.models.spade_models."""

import itertools

from absl.testing import parameterized
from pathdreamer import constants
from pathdreamer.models import spade_models
import tensorflow as tf


class LayersTest(tf.test.TestCase, parameterized.TestCase):
  """Tests on the image_models file."""

  @parameterized.parameters(list(itertools.product(
      (2,), (128, 256), (3,), (True, False), (3, 8), (16, 32))))
  def test_multispade(self, batch_size, image_size, kernel_size, circular_pad,
                      input_dim, output_dim):
    """Tests that VGGLayer outputs correct shapes."""
    test_input = tf.random.uniform(
        (batch_size, image_size, image_size, input_dim), 0, 1, dtype=tf.float32)
    test_guidance = tf.random.uniform(
        (batch_size, image_size, image_size, 3), 0, 1, dtype=tf.float32)
    test_guidance_mask = tf.random.uniform(
        (batch_size, image_size, image_size, 1), 0, 1, dtype=tf.float32)
    # Cast to a valid mask.
    test_guidance_mask = tf.cast(test_guidance_mask < 0.5, tf.float32)
    test_guidance = test_guidance * test_guidance_mask
    test_segmentation = tf.random.uniform((batch_size, image_size, image_size),
                                          0,
                                          constants.NUM_MP3D_CLASSES,
                                          dtype=tf.int32)
    test_segmentation = tf.one_hot(
        test_segmentation, constants.NUM_MP3D_CLASSES)

    # Test multispade layer.
    test_layer = spade_models.MultiSPADE(32, kernel_size, circular_pad)
    layer_output = test_layer(test_input, test_segmentation, test_guidance,
                              test_guidance_mask)
    self.assertEqual(layer_output.shape, test_input.shape)

    # Test multispade resblock.
    test_model = spade_models.MultiSPADEResBlock(input_dim, output_dim,
                                                 circular_pad)
    model_output = test_model(test_input, test_segmentation, test_guidance,
                              test_guidance_mask)
    self.assertEqual(model_output.shape, test_input.shape[:3] + (output_dim,))

  @parameterized.parameters(list(itertools.product(
      (2,), (128, 256), (True, False))))
  def test_wcspade_generator(self, batch_size, image_size, circular_pad):
    """Tests that VGGLayer outputs correct shapes."""
    height, width = image_size, image_size * 2
    test_depth = tf.random.uniform(
        (batch_size, height, width, 1), 0, 1, dtype=tf.float32)
    test_guidance = tf.random.uniform(
        (batch_size, height, width, 3), 0, 1, dtype=tf.float32)
    test_guidance_mask = tf.random.uniform(
        (batch_size, height, width, 1), 0, 1, dtype=tf.float32)
    # Cast to a valid mask.
    test_guidance_mask = tf.cast(test_guidance_mask < 0.5, tf.float32)
    test_guidance = test_guidance * test_guidance_mask
    test_segmentation = tf.random.uniform((batch_size, height, width),
                                          0,
                                          constants.NUM_MP3D_CLASSES,
                                          dtype=tf.int32)
    test_segmentation = tf.one_hot(test_segmentation,
                                   constants.NUM_MP3D_CLASSES)
    inputs = {
        'one_hot_mask': test_segmentation,
        'depth': test_depth,
        'proj_image': test_guidance,
        'proj_mask': test_guidance_mask,
        'image': test_guidance,
        'prev_image': tf.zeros_like(test_guidance),
        'first_frame': tf.zeros((batch_size,))
    }

    # Test multispade layer.
    test_model = spade_models.WCSPADEGenerator(
        batch_size, image_size=image_size, circular_pad=circular_pad)
    _, _, model_output = test_model([inputs, None])
    self.assertEqual(
        model_output.shape, (batch_size, height, width, 3))
    self.assertAllInRange(model_output, 0, 1)

  @parameterized.parameters(list(itertools.product(
      (3, 4), (2, 3), (True, False))))
  def test_discriminator(self, kernel_size, n_layers, circular_pad):
    test_input = tf.random.uniform((2, 256, 512, 3), 0, 1, dtype=tf.float32)
    discriminator = spade_models.SNPatchDiscriminator(
        kernel_size=kernel_size, n_layers=n_layers, circular_pad=circular_pad)
    out = discriminator(test_input)
    # Output should have n_layers + final true/false prediction layer.
    self.assertLen(out, n_layers + 1)
    # Last layer should output dimension 1 tensor indicating real/fake.
    self.assertEqual(out[-1].shape[-1], 1)

  @parameterized.parameters(list(itertools.product(
      (3, 4), (1, 2), (2,), (True, False))))
  def test_multiscale_discriminator(self, kernel_size, n_dis, n_layers,
                                    circular_pad):
    test_input = tf.random.uniform((2, 256, 512, 3), 0, 1, dtype=tf.float32)
    discriminator = spade_models.SNMultiScaleDiscriminator(
        n_dis=n_dis, kernel_size=kernel_size, n_layers=n_layers,
        circular_pad=circular_pad)
    out = discriminator(test_input)
    # Output should have one list for each discriminator.
    self.assertLen(out, n_dis)
    for sub_out in out:
      self.assertIsInstance(sub_out, list)
      # Each sublist is the output of an independent SNPatchDiscriminator.
      self.assertLen(sub_out, n_layers + 1)
      self.assertEqual(sub_out[-1].shape[-1], 1)


if __name__ == '__main__':
  tf.test.main()
