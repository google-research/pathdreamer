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

"""Tests for pathdreamer.utils.utils."""

from absl.testing import parameterized
import tensorflow as tf

from pathdreamer.utils import utils


class UtilsTest(tf.test.TestCase, parameterized.TestCase):
  """Tests on the image_models file."""

  @parameterized.parameters((1, 5, 128), (2, 3, 256))
  def test_compute_sequence_iou(self, batch_size, seq_len, image_size):
    """Tests that the sequence IOU returns valid values."""
    num_classes = 41
    test_pred = tf.random.uniform((batch_size, seq_len, image_size, image_size),
                                  maxval=num_classes, dtype=tf.int32)
    test_pred_one_hot = tf.cast(tf.one_hot(test_pred, num_classes), tf.float32)
    test_gt = tf.random.uniform((batch_size, seq_len, image_size, image_size),
                                maxval=num_classes, dtype=tf.int32)
    test_gt_one_hot = tf.cast(tf.one_hot(test_gt, num_classes), tf.float32)
    test_mask = tf.random.uniform(
        (batch_size, seq_len), maxval=1, dtype=tf.int32)
    test_mask = tf.cast(test_mask, tf.float32)
    seq_iou, miou = utils.compute_sequence_iou(
        test_pred_one_hot, test_gt_one_hot, test_mask)
    self.assertAllGreaterEqual(seq_iou, 0)
    self.assertEqual(seq_iou.shape, (batch_size, seq_len))
    self.assertGreaterEqual(miou, 0)

  def test_iou_zero_mask(self):
    """Tests that the sequence IOU returns 0s with an empty mask."""
    batch_size, seq_len, image_size, num_classes = 2, 5, 128, 41
    test_pred = tf.random.uniform((batch_size, seq_len, image_size, image_size),
                                  maxval=num_classes, dtype=tf.int32)
    test_pred_one_hot = tf.cast(tf.one_hot(test_pred, num_classes), tf.float32)
    test_mask = tf.zeros((batch_size, seq_len))
    seq_iou, miou = utils.compute_sequence_iou(
        test_pred_one_hot, test_pred_one_hot, test_mask)
    self.assertAllClose(seq_iou, test_mask)  # Check that everything is 0.
    self.assertAlmostEqual(miou.numpy(), 0)

  @parameterized.parameters((1, 32, 16), (2, 16, 8))
  def test_compute_kld(self, batch_size, image_size, dims):
    """Tests that the KLD function returns valid values."""
    test_mu1 = tf.random.normal((batch_size, image_size, image_size, dims))
    test_logvar1 = tf.random.normal((batch_size, image_size, image_size, dims))
    test_mu2 = tf.random.normal((batch_size, image_size, image_size, dims))
    test_logvar2 = tf.random.normal((batch_size, image_size, image_size, dims))
    kld = utils.compute_kl(test_mu1, test_logvar1, test_mu2, test_logvar2)
    self.assertEqual(kld.shape, test_mu1.shape)
    self.assertAllGreaterEqual(kld, 0)

    # Test that KLD is equal for the same distribution.
    kld_same = utils.compute_kl(test_mu1, test_logvar1, test_mu1, test_logvar1)
    self.assertAllEqual(kld_same, tf.zeros_like(kld_same))


if __name__ == '__main__':
  tf.test.main()
