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

"""SPADE models."""

from typing import List, Tuple

from pathdreamer.models import layers
import tensorflow as tf
from tensorflow_addons import layers as tfa_layers


def reparameterize(mu: tf.Tensor, logvar: tf.Tensor) -> tf.Tensor:
  """Reparameterization trick to draw random vector from N(mu, sigma)."""
  sigma = tf.math.exp(0.5 * logvar)
  eps = tf.random.normal(sigma.shape, dtype=mu.dtype)
  return eps * sigma + mu


class MultiSPADE(tf.keras.layers.Layer):
  """Spatially-Adaptive Normalization layer with guidance images.

  Adapted from World-Consistent Video-to-Video Synthesis
  (https://arxiv.org/abs/2007.08509).
  """

  def __init__(self,
               hidden_dims: int = 128,
               kernel_size: int = 3,
               circular_pad: bool = False):
    """Initializes SPADE normalization layer.

    Args:
      hidden_dims: Dimensions of the intermediate embedding layer.
      kernel_size: Kernel size of conv layers.
      circular_pad: Whether to apply circular pad.
    """
    super().__init__()
    self.hidden_dims = hidden_dims
    self.kernel_size = kernel_size
    self.circular_pad = circular_pad

  def build(self, input_shape):
    input_dims = input_shape[-1]
    self.param_free_norm = tf.keras.layers.experimental.SyncBatchNormalization(
        center=False, scale=False)

    self.map_conv = tf.keras.Sequential([
        layers.PadLayer(
            padding=self.kernel_size // 2, circular_pad=self.circular_pad),
        tf.keras.layers.Conv2D(
            self.hidden_dims, self.kernel_size, padding='valid'),
        tf.keras.layers.ReLU(),
    ])
    self.gamma_conv = tf.keras.Sequential([
        layers.PadLayer(
            padding=self.kernel_size // 2, circular_pad=self.circular_pad),
        tf.keras.layers.Conv2D(input_dims, self.kernel_size, padding='valid')
    ])
    self.beta_conv = tf.keras.Sequential([
        layers.PadLayer(
            padding=self.kernel_size // 2, circular_pad=self.circular_pad),
        tf.keras.layers.Conv2D(input_dims, self.kernel_size, padding='valid')
    ])

    self.guidance_pad = layers.PadLayer(
        padding=self.kernel_size // 2, circular_pad=self.circular_pad)
    self.guidance_conv = layers.PartialConv(
        self.hidden_dims, self.kernel_size, padding='valid')
    self.relu = tf.keras.layers.ReLU()

    self.guidance_gamma_conv = tf.keras.Sequential([
        layers.PadLayer(
            padding=self.kernel_size // 2, circular_pad=self.circular_pad),
        tf.keras.layers.Conv2D(input_dims, self.kernel_size, padding='valid')
    ])
    self.guidance_beta_conv = tf.keras.Sequential([
        layers.PadLayer(
            padding=self.kernel_size // 2, circular_pad=self.circular_pad),
        tf.keras.layers.Conv2D(input_dims, self.kernel_size, padding='valid')
    ])

  def call(self, x: tf.Tensor, segmentation_map: tf.Tensor,
           guidance_image: tf.Tensor, guidance_mask: tf.Tensor) -> tf.Tensor:
    """Forward pass of MultiSPADE model.

    Args:
      x: Input tensor of shape (N, H, W, C).
      segmentation_map: One hot tensor of shape (N, H, W, D).
      guidance_image: RGB tensor of shape (N, H, W, 3) with values from [0, 1].
      guidance_mask: Binary tensor of shape (N, H, W, 1).
    Returns:
      out: (N, H, W, C) output tensor.
    """
    # Unconditional normalization.
    normed_x = self.param_free_norm(x)

    # Compute scaling / bias conditioned on the segmentation map.
    scaled_map = tf.image.resize(
        segmentation_map, (x.shape[1], x.shape[2]), method='nearest')
    map_out = self.map_conv(scaled_map)
    gamma = self.gamma_conv(map_out)
    beta = self.beta_conv(map_out)
    out = normed_x * (1 + gamma) + beta

    scaled_guidance_image = tf.image.resize(
        guidance_image, (x.shape[1], x.shape[2]), method='nearest')
    scaled_guidance_mask = tf.image.resize(
        guidance_mask, (x.shape[1], x.shape[2]), method='nearest')
    scaled_guidance_image = self.guidance_pad(scaled_guidance_image)
    scaled_guidance_mask = self.guidance_pad(scaled_guidance_mask)
    guidance_out, _ = self.guidance_conv(scaled_guidance_image,
                                         scaled_guidance_mask)
    guidance_out = self.relu(guidance_out)
    guidance_gamma = self.guidance_gamma_conv(guidance_out)
    guidance_beta = self.guidance_beta_conv(guidance_out)
    out = out * (1 + guidance_gamma) + guidance_beta

    return out


class MultiSPADEResBlock(tf.keras.Model):
  """MultiSPADE normalized residual block."""

  def __init__(self, input_dim: int, output_dim: int, circular_pad: bool):
    """Initializes a MultiSPADE normalized ResBlock.

    Args:
      input_dim: Input dimensions of tensor.
      output_dim: Output dimensions after passing through this block.
      circular_pad: Whether to apply circular padding.
    """
    super().__init__()

    hidden_dim = min(input_dim, output_dim)

    self.conv1 = tf.keras.Sequential([
        layers.PadLayer(padding=1, circular_pad=circular_pad),
        layers.SpectralConv(
            hidden_dim,
            kernel_size=3,
            strides=1,
            padding='VALID',
            activation=None)
    ])
    self.conv2 = tf.keras.Sequential([
        layers.PadLayer(padding=1, circular_pad=circular_pad),
        layers.SpectralConv(
            output_dim,
            kernel_size=3,
            strides=1,
            padding='VALID',
            activation=None)
    ])

    # Normalization layers.
    self.norm1 = MultiSPADE(circular_pad=circular_pad)
    self.norm2 = MultiSPADE(circular_pad=circular_pad)

    self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.2)

    # If input and output dims are equal, shortcut is an identity mapping.
    if input_dim == output_dim:
      self.shortcut = None
    else:
      self.shortcut = layers.SpectralConv(
          output_dim,
          kernel_size=1,
          strides=1,
          padding='SAME',
          activation=None,
          use_bias=False)
      self.shortcut_norm = MultiSPADE(circular_pad=circular_pad)

  def call(self, x: tf.Tensor, segmentation_map: tf.Tensor,
           guidance_image: tf.Tensor, guidance_mask: tf.Tensor) -> tf.Tensor:
    if self.shortcut is None:
      shortcut_out = x
    else:
      shortcut_norm = self.shortcut_norm(x, segmentation_map, guidance_image,
                                         guidance_mask)
      shortcut_out = self.shortcut(shortcut_norm)

    x_out = self.norm1(x, segmentation_map, guidance_image, guidance_mask)
    x_out = self.leaky_relu(x_out)
    x_out = self.conv1(x_out)

    x_out = self.norm2(x_out, segmentation_map, guidance_image, guidance_mask)
    x_out = self.leaky_relu(x_out)
    x_out = self.conv2(x_out)

    return shortcut_out + x_out


class WCSPADEGenerator(tf.keras.Model):
  """World Consistent SPADE generator model.

  This replaces the SPADE layers in the original SPADE model with the
  multi-SPADE layers proposed in https://arxiv.org/abs/2007.08509.
  """

  def __init__(self,
               batch_size: int,
               image_size: int = 256,
               gen_dims: int = 96,
               z_dim: int = 256,
               circular_pad: bool = True,
               use_depth_condition: bool = True,
               use_seg_condition: bool = True,
               fully_conv: bool = True):
    """Initializes a SPADE generator.

    Args:
      batch_size: Batch size of inputs.
      image_size: Height and width in pixels of desired output.
      gen_dims: Multiplier for dimensions in hidden layer.
      z_dim: Dimensions of noise vector.
      circular_pad: Whether to apply circular padding.
      use_depth_condition: If True, includes depth inputs as condition to the
        model. Either this or use_seg_condition (or both) should be true.
      use_seg_condition: If True, includes segmentation inputs as condition to
        the model. Either this or use_seg_condition (or both) should be true.
      fully_conv: Whether to use fully convolutional network.
    """
    super().__init__()
    self.batch_size = batch_size
    self.z_dim = z_dim
    self.num_upsample_layers = 5
    self.use_depth_condition = use_depth_condition
    self.use_seg_condition = use_seg_condition
    assert self.use_depth_condition or self.use_seg_condition
    self.fully_conv = fully_conv

    self.block1 = MultiSPADEResBlock(
        16 * gen_dims, 16 * gen_dims, circular_pad=circular_pad)
    self.block2 = MultiSPADEResBlock(
        16 * gen_dims, 16 * gen_dims, circular_pad=circular_pad)
    self.block3 = MultiSPADEResBlock(
        16 * gen_dims, 16 * gen_dims, circular_pad=circular_pad)
    self.block4 = MultiSPADEResBlock(
        16 * gen_dims, 8 * gen_dims, circular_pad=circular_pad)
    self.block5 = MultiSPADEResBlock(
        8 * gen_dims, 4 * gen_dims, circular_pad=circular_pad)
    self.block6 = MultiSPADEResBlock(
        4 * gen_dims, 2 * gen_dims, circular_pad=circular_pad)
    self.block7 = MultiSPADEResBlock(
        2 * gen_dims, 1 * gen_dims, circular_pad=circular_pad)

    self.img_fc = tf.keras.Sequential([
        layers.PadLayer(padding=1, circular_pad=circular_pad),
        tf.keras.layers.Conv2D(3, kernel_size=3, padding='valid')
    ])
    self.upsample = tf.keras.layers.UpSampling2D()
    self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.2)

    # Width / height of random noise or segmentation map.
    self.output_size = image_size
    self.input_size = image_size // (2**self.num_upsample_layers)
    self.fc = tf.keras.layers.Conv2D(
        16 * gen_dims, kernel_size=3, padding='same')

  def call(self, inputs, training=None) -> List[tf.Tensor]:
    """Performs a forward pass to generate an image.

    Args:
      inputs: List of two inputs. The first item is the condition, the second is
        the random noise (if any).
      training: Whether the model is in training mode.

    Returns:
      out_x: Tensor of shape (N, H, W, 3) with values in [0, 1] representing a
        generated image.
    """
    cond, z = inputs
    del z
    _, image_height, _, _ = cond['one_hot_mask'].shape
    input_image_height = image_height // (2**self.num_upsample_layers)
    if self.use_depth_condition and self.use_seg_condition:
      x = tf.concat([cond['one_hot_mask'], cond['depth']], axis=-1)
    elif self.use_depth_condition:
      x = cond['depth']
    elif self.use_seg_condition:
      x = cond['one_hot_mask']
    prev_frame = cond['prev_image']
    guidance_image, guidance_mask = cond['proj_image'], cond['proj_mask']

    # Unused.
    mu = tf.zeros((x.shape[0], self.z_dim))
    logvar = tf.zeros((x.shape[0], self.z_dim))

    # Use image encoder if a previous frame exists
    downsampled_x = tf.image.resize(
        prev_frame, (input_image_height, input_image_height * 2),
        method='nearest')
    out_x = self.fc(downsampled_x)

    out_x = self.block1(out_x, x, guidance_image, guidance_mask)
    out_x = self.upsample(out_x)
    out_x = self.block2(out_x, x, guidance_image, guidance_mask)
    out_x = self.block3(out_x, x, guidance_image, guidance_mask)
    out_x = self.upsample(out_x)
    out_x = self.block4(out_x, x, guidance_image, guidance_mask)
    out_x = self.upsample(out_x)
    out_x = self.block5(out_x, x, guidance_image, guidance_mask)
    out_x = self.upsample(out_x)
    out_x = self.block6(out_x, x, guidance_image, guidance_mask)
    out_x = self.upsample(out_x)
    out_x = self.block7(out_x, x, guidance_image, guidance_mask)
    out_x = self.leaky_relu(out_x)

    out_x = self.img_fc(out_x)
    out_x = tf.math.tanh(out_x)

    # Cast to [0, 1]
    out_x = (out_x + 1) / 2
    return [mu, logvar, out_x]


class SNPatchDiscriminator(tf.keras.Model):
  """Spectral normalized PatchGAN discriminator."""

  def __init__(self,
               kernel_size: int = 4,
               dis_dims: int = 64,
               n_layers: int = 4,
               circular_pad: bool = False):
    """Initializes a spectral normalized PatchGAN discriminator.

    Args:
      kernel_size: Kernel size of convolutions.
      dis_dims: Baseline dimensions for convolutional layers.
      n_layers: Number of layers in this discriminator.
      circular_pad: Whether to apply circular padding.
    """
    super().__init__()

    self.discriminator_groups = [
        tf.keras.Sequential([
            layers.PadLayer(kernel_size // 2, circular_pad=circular_pad),
            tf.keras.layers.Conv2D(
                dis_dims,
                kernel_size=kernel_size,
                strides=2,
                padding='VALID'),
            tf.keras.layers.LeakyReLU(alpha=0.2)
        ])
    ]

    previous_dim = dis_dims
    for i in range(1, n_layers):
      current_dim = min(previous_dim * 2, 512)
      self.discriminator_groups.append(
          tf.keras.Sequential([
              layers.PadLayer(kernel_size // 2, circular_pad=circular_pad),
              layers.SpectralConv(
                  current_dim,
                  kernel_size=kernel_size,
                  strides=2 if (i != n_layers-1) else 1,
                  padding='VALID',
                  activation=None),
              tfa_layers.InstanceNormalization(),
              tf.keras.layers.LeakyReLU(alpha=0.2),
          ])
      )
      previous_dim = current_dim

    # Final classification layer
    self.discriminator_groups.append(
        tf.keras.layers.Conv2D(
            1, kernel_size=kernel_size, strides=1, padding='SAME'))

  def call(self, x: tf.Tensor) -> List[tf.Tensor]:
    """Forward pass of PatchGAN discriminator.

    Args:
      x: Tensor of shape (N, H, W, C).

    Returns:
      results: List of Tensors of intermediate features. Each object in the list
        is of shape (N, H, W, C).
    """
    results = []
    prev_out = x
    for model in self.discriminator_groups:
      out = model(prev_out)
      results.append(out)
      prev_out = out
    return results


class SNMultiScaleDiscriminator(tf.keras.Model):
  """Spectral normalized multiscale PatchGAN discriminator.

  Modified from "Semantic Image Synthesis with Spatially-Adaptive Normalization"
  (https://arxiv.org/abs/1903.07291)
  """

  def __init__(self,
               image_size: int = 256,
               n_dis: int = 2,
               kernel_size: int = 4,
               dis_dims: int = 96,
               n_layers: int = 5,
               circular_pad: bool = False):
    """Initializes a spectral normalized PatchGAN discriminator.

    Args:
      image_size: Size of image inputs. This is not used in this model as it is
        fully convolutional, but remains here for compatibility with the
        trainer.
      n_dis: Number of discriminators to use.
      kernel_size: Kernel size of convolutions in sub-discriminators.
      dis_dims: Baseline dims for convolutional layers in sub-discriminators.
      n_layers: Number of layers in each sub-discriminator.
      circular_pad: Whether to apply circular padding.
    """
    super().__init__()
    del image_size  # Not used - this model is fully convolutional.

    self.discriminators = []
    for _ in range(n_dis):
      self.discriminators.append(
          SNPatchDiscriminator(
              kernel_size=kernel_size,
              dis_dims=dis_dims,
              n_layers=n_layers,
              circular_pad=circular_pad))

  def call(self, inputs: tf.Tensor) -> List[List[tf.Tensor]]:
    """Forward pass of multiscale discriminator.

    Args:
      inputs: Tensor of shape (N, H, W, C).

    Returns:
      results: List of list of Tensors of intermediate features. The first
        Tensor in the list has shape (N, H', W', C'), and each Tensor after is
        downsampled by a factor of 2, e.g. (N, H'//2, W'//2, C'), and so on.
    """
    result = []
    prev_out = inputs
    for model in self.discriminators:
      out = model(prev_out)
      result.append(out)

      # Downsample for next discriminator.
      prev_out = tf.nn.avg_pool(prev_out, ksize=3, strides=2, padding='SAME')
    return result
