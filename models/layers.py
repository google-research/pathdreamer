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

"""Common layers used for modeling."""

from typing import Optional, Tuple
import tensorflow as tf


class PadLayer(tf.keras.layers.Layer):
  """Implements circular and regular padding."""

  def __init__(self, padding: int, circular_pad: bool = False, **kwargs):
    """Instantiates a PadLayer.

    Args:
      padding: Size of padding in pixels.
      circular_pad: If true, uses circular padding along the width dimension.
      **kwargs: Additional arguments passed to tf.keras.layers.Layer.
    """
    super().__init__(**kwargs)
    self.padding = padding
    self.circular_pad = circular_pad

  def call(self, inputs: tf.Tensor, training=None) -> tf.Tensor:
    """Implements forward pass for padding.

    Args:
      inputs: tf.Tensor input of shape (N, H, W, C).
      training: Whether the layer is in training mode.

    Returns:
      tf.Tensor, the normalized output.
    """
    batch_size, height, width, channels = inputs.shape
    left_pad = tf.zeros((batch_size, height, self.padding, channels),
                        dtype=inputs.dtype)
    right_pad = tf.zeros((batch_size, height, self.padding, channels),
                         dtype=inputs.dtype)
    if self.circular_pad:
      left_pad = inputs[:, :, -self.padding:, :]
      right_pad = inputs[:, :, :self.padding, :]
    top_pad = tf.zeros(
        (batch_size, self.padding, width + self.padding * 2, channels),
        dtype=inputs.dtype)
    bottom_pad = tf.zeros(
        (batch_size, self.padding, width + self.padding * 2, channels),
        dtype=inputs.dtype)
    padded_tensor = tf.concat([left_pad, inputs, right_pad], axis=2)
    padded_tensor = tf.concat([bottom_pad, padded_tensor, top_pad], axis=1)
    return padded_tensor


class Bottleneck(tf.keras.Model):
  """ResNet bottleneck block."""

  def __init__(self,
               filters: int = 128,
               strides: int = 1,
               expansion: int = 4,
               downsample=None,
               circular_pad: bool = False):
    super(Bottleneck, self).__init__()
    self.shortcut = None
    self.main = tf.keras.Sequential([
        tf.keras.layers.Conv2D(
            filters, kernel_size=1, strides=1, padding='SAME'),
        tf.keras.layers.experimental.SyncBatchNormalization(),
        tf.keras.layers.ReLU(),
        PadLayer(1, circular_pad=circular_pad),
        tf.keras.layers.Conv2D(
            filters, kernel_size=3, strides=strides, padding='VALID'),
        tf.keras.layers.experimental.SyncBatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(
            expansion * filters, kernel_size=1, strides=1, padding='SAME'),
        tf.keras.layers.experimental.SyncBatchNormalization(),
    ])
    self.relu = tf.keras.layers.ReLU()
    self.downsample = downsample

  def call(self, x: tf.Tensor, training=None) -> tf.Tensor:
    residual = x
    out = self.main(x)
    if self.downsample is not None:
      residual = self.downsample(x)
    out += residual
    out = self.relu(out)
    return out


class SpectralConv(tf.keras.layers.Conv2D):
  """Convolution with spectral normalization applied to weights.

  From "Spectral Normalization for Generative Adversarial Networks"
  https://arxiv.org/abs/1802.05957
  """

  def build(self, input_shape):
    was_built = self.built
    tf.keras.layers.Conv2D.build(self, input_shape)
    self.built = was_built
    output_dims = self.kernel.shape[-1]
    self.u = self.add_weight(
        name=self.name + '_u',
        shape=[1, output_dims],
        dtype=tf.float32,
        initializer=tf.initializers.TruncatedNormal(),
        trainable=False,
        aggregation=tf.VariableAggregation.MEAN)

    if not isinstance(self.padding, (list, tuple)):
      self.padding = self.padding.upper()

    self.built = True

  def call(self, feature, training=None):
    """Forward pass applying spectral normalized convolution.

    Args:
      feature: Float tensor of shape (N, H, W, C), representing input feature.
      training: Represents whether the layer is in training mode.

    Returns:
      out: Float tensor of shape (N, H, W, output_dims), representing output
        feature after applying a spectral normalized convolution.
    """
    # For preventing division by 0.
    eps = 1e-10
    # Flatten weight matrix.
    w_shape = self.kernel.shape
    w = tf.reshape(self.kernel, [-1, w_shape[-1]])

    # One step of power iteration.
    v = tf.matmul(self.u, w, transpose_b=True)
    v_hat = v / (tf.norm(v) + eps)

    u = tf.matmul(v_hat, w)
    u_hat = u / (tf.norm(u) + eps)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), u_hat, transpose_b=True)

    if training:
      self.u.assign(u_hat)
    w_norm = w / (sigma + eps)
    w_norm = tf.reshape(w_norm, w_shape)

    out = tf.nn.conv2d(
        input=feature,
        filters=w_norm,
        strides=self.strides,
        dilations=self.dilation_rate,
        padding=self.padding)

    if self.use_bias:
      out = out + self.bias

    if self.activation:
      out = self.activation(out)

    return out


class PartialConv(tf.keras.layers.Conv2D):
  """Partial 2D convolution.

  From "Image inpainting for irregular holes using partial convolutions.",
    Liu et al., ECCV 2018.
  """

  def build(self, input_shape):
    was_built = self.built
    tf.keras.layers.Conv2D.build(self, input_shape)
    self.built = was_built
    ks_height, ks_width, _, _ = self.kernel.shape
    self.weight_mask_updater = tf.ones((ks_height, ks_width, 1, 1))
    self.slide_window_size = ks_height * ks_width * 1
    self.built = True

  def call(self,
           feature: tf.Tensor,
           mask: Optional[tf.Tensor] = None,
           training=None) -> Tuple[tf.Tensor, tf.Tensor]:
    """Forward pass applying partial convolution.

    Args:
      feature: Float tensor of shape (N, H, W, C) representing input feature.
      mask: Binary float tensor of shape (N, H, W, 1) representing valid pixels.
      training: Represents whether the layer is in training mode.

    Returns:
      out: Float tensor of shape (N, H, W, output_dims), representing output
        feature after applying a partial convolution.
    """
    if mask is None:
      mask = tf.ones((feature.shape[0], feature.shape[1], feature.shape[2], 1))
    eps = 1e-6
    update_mask = tf.nn.conv2d(
        mask,
        self.weight_mask_updater,
        strides=self.strides,
        padding=self.padding.upper())
    mask_ratio = self.slide_window_size / (update_mask + eps)
    update_mask = tf.clip_by_value(update_mask, 0, 1)
    mask_ratio = mask_ratio * update_mask
    mask = tf.stop_gradient(mask)
    update_mask = tf.stop_gradient(update_mask)
    mask_ratio = tf.stop_gradient(mask_ratio)

    out = feature * mask
    out = tf.nn.conv2d(
        input=out,
        filters=self.kernel,
        strides=self.strides,
        padding=self.padding.upper())
    if self.bias is not None:
      bias = tf.reshape(self.bias, (1, 1, 1, -1))
      out = (out - bias) * mask_ratio + bias
      out = out * update_mask
    else:
      out = out * mask_ratio
    return out, update_mask


class ResStack(tf.keras.Model):
  """Single ResNet stack consisting of multiple Bottleneck blocks."""

  def __init__(self,
               inplanes: int,
               planes: int,
               blocks: int,
               strides: int = 1,
               expansion: int = 4,
               circular_pad: bool = False):
    super(ResStack, self).__init__()
    downsample = None
    if strides != 1 or inplanes != planes * expansion:
      downsample = tf.keras.Sequential([
          tf.keras.layers.Conv2D(
              planes * expansion,
              kernel_size=1,
              strides=strides,
              padding='SAME',
              use_bias=False),
          tf.keras.layers.experimental.SyncBatchNormalization()
      ])
    block_models = [
        Bottleneck(
            planes,
            strides=strides,
            expansion=expansion,
            downsample=downsample,
            circular_pad=circular_pad)
    ]
    for _ in range(blocks - 1):
      block_models.append(
          Bottleneck(planes, expansion=expansion, circular_pad=circular_pad))
    self.block = tf.keras.Sequential(block_models)

  def call(self, x: tf.Tensor, training=None) -> tf.Tensor:
    return self.block(x)


class TransBasicBlock(tf.keras.Model):
  """Bottleneck block with transposed convolutions.

  This block performs upsampling if required.
  """

  def __init__(self,
               inplanes: int,
               planes: int,
               blocks: int,
               strides: int = 1,
               upsample=None,
               circular_pad: bool = False):
    super(TransBasicBlock, self).__init__()
    conv2 = None
    if upsample is not None and strides != 1:
      conv2 = tf.keras.layers.Conv2DTranspose(
          planes,
          kernel_size=3,
          strides=strides,
          padding='SAME',
          output_padding=1,
          use_bias=False)
    else:
      conv2 = tf.keras.Sequential([
          PadLayer(1, circular_pad=circular_pad),
          tf.keras.layers.Conv2D(
              planes,
              kernel_size=3,
              strides=strides,
              padding='VALID',
              use_bias=False)
      ])

    self.main = tf.keras.Sequential([
        PadLayer(1, circular_pad=circular_pad),
        tf.keras.layers.Conv2D(
            inplanes, kernel_size=3, strides=1, padding='VALID',
            use_bias=False),
        tf.keras.layers.experimental.SyncBatchNormalization(),
        tf.keras.layers.ReLU(),
        conv2,
        tf.keras.layers.experimental.SyncBatchNormalization(),
    ])
    self.upsample = upsample
    self.relu = tf.keras.layers.ReLU()

  def call(self, x: tf.Tensor, training=None) -> tf.Tensor:
    residual = x
    out_x = self.main(x)
    if self.upsample is not None:
      residual = self.upsample(x)
    out_x += residual
    out_x = self.relu(out_x)
    return out_x


class ResStackTranspose(tf.keras.Model):
  """ResNet stack consisting of transposed blocks.

  This stack performs upsampling if required (if strides > 1).
  """

  def __init__(self,
               inplanes: int,
               planes: int,
               blocks: int,
               strides: int = 1,
               circular_pad: bool = False):
    super(ResStackTranspose, self).__init__()
    upsample = None
    if strides != 1:
      upsample = tf.keras.Sequential([
          tf.keras.layers.Conv2DTranspose(
              planes,
              kernel_size=2,
              strides=strides,
              padding='VALID',
              use_bias=False),
          tf.keras.layers.experimental.SyncBatchNormalization()
      ])
    elif inplanes != planes:
      upsample = tf.keras.Sequential([
          tf.keras.layers.Conv2D(
              planes, kernel_size=1, strides=strides, use_bias=False),
          tf.keras.layers.experimental.SyncBatchNormalization()
      ])

    block_models = []
    for _ in range(blocks - 1):
      block_models.append(
          TransBasicBlock(
              inplanes, inplanes, blocks, circular_pad=circular_pad))
    block_models += [
        TransBasicBlock(
            inplanes,
            planes,
            blocks,
            strides,
            upsample=upsample,
            circular_pad=circular_pad)
    ]
    self.block = tf.keras.Sequential(block_models)

  def call(self, x: tf.Tensor, training=None) -> tf.Tensor:
    return self.block(x)
