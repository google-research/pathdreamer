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

"""Image models used in structure / video prediction."""

from typing import List, Tuple

from pathdreamer.models import layers
import tensorflow as tf


def _check_image_size(image_size):
  """Check for valid image sizes."""
  valid_image_sizes = [128, 256]
  if image_size not in valid_image_sizes:
    raise ValueError(f'image_size should be one of {valid_image_sizes}.')


class ResNetEncoder(tf.keras.Model):
  """Encoder architecture for ResNet image model.

  Modified from "RedNet: Residual Encoder-Decoder Network for indoor RGB-D
    Semantic Segmentation": https://arxiv.org/abs/1806.01054"
  """

  def __init__(self,
               image_size: int,
               hidden_dims: int = 64,
               resnet_version: str = '50',  # either 50, 101, or 152
               flatten_output: bool = True,
               circular_pad: bool = False):
    super(ResNetEncoder, self).__init__()

    # If model is not fully convolutional, check that image size is valid.
    if flatten_output:
      _check_image_size(image_size)
    self.image_size = image_size
    self.flatten_output = flatten_output

    self.block1 = tf.keras.Sequential([
        layers.PadLayer(3, circular_pad=circular_pad),
        tf.keras.layers.Conv2D(hidden_dims, 7, strides=2, padding='VALID'),
        tf.keras.layers.experimental.SyncBatchNormalization(),
        tf.keras.layers.ReLU()
    ])

    if resnet_version == '50':
      filters = [3, 4, 6, 3]
    elif resnet_version == '101':
      filters = [3, 4, 23, 3]
    elif resnet_version == '152':
      filters = [3, 8, 36, 3]
    else:
      raise ValueError('resnet_version should be one of ["50", "101", "152"], '
                       f'got {resnet_version} instead.')

    self.stack1 = layers.ResStack(
        hidden_dims, hidden_dims, filters[0], circular_pad=circular_pad)
    self.stack2 = layers.ResStack(
        hidden_dims, hidden_dims * 2, filters[1], strides=2,
        circular_pad=circular_pad)
    self.stack3 = layers.ResStack(
        hidden_dims * 2,
        hidden_dims * 4,
        filters[2],
        strides=2,
        circular_pad=circular_pad)
    self.stack4 = layers.ResStack(
        hidden_dims * 4,
        hidden_dims * 8,
        filters[3],
        strides=2,
        circular_pad=circular_pad)
    self.flatten = tf.keras.layers.Flatten()
    self.maxpool = tf.keras.Sequential(
        [tf.keras.layers.MaxPool2D(padding='SAME')])
    self.final_conv = tf.keras.Sequential([
        layers.PadLayer(1, circular_pad=circular_pad),
        tf.keras.layers.Conv2D(
            hidden_dims, kernel_size=3, strides=1, padding='VALID'),
        tf.keras.layers.experimental.SyncBatchNormalization(),
        tf.keras.layers.ReLU(),
    ])

  def call(self,
           x: tf.Tensor,
           training=None) -> Tuple[tf.Tensor, List[tf.Tensor]]:
    out_x = self.block1(x)
    b1 = out_x
    out_x = self.maxpool(out_x)
    out_x = self.stack1(out_x)
    s1 = out_x
    out_x = self.stack2(out_x)
    s2 = out_x
    out_x = self.stack3(out_x)
    s3 = out_x
    out_x = self.stack4(out_x)
    out_x = self.final_conv(out_x)
    if self.flatten_output:
      out_x = self.flatten(out_x)
    return out_x, [b1, s1, s2, s3]


class ResNetDecoder(tf.keras.Model):
  """Decoder architecture for ResNet image model.

  Modified from "RedNet: Residual Encoder-Decoder Network for indoor RGB-D
    Semantic Segmentation": https://arxiv.org/abs/1806.01054"
  """

  def create_agent(self, hidden_dims: int):
    agent = tf.keras.Sequential([
        tf.keras.layers.Conv2D(
            hidden_dims,
            kernel_size=1,
            strides=1,
            padding='SAME',
            use_bias=False),
        tf.keras.layers.experimental.SyncBatchNormalization(),
        tf.keras.layers.ReLU()
    ])
    return agent

  def __init__(self,
               output_dim: int,
               image_size: int,
               hidden_dims: int = 64,
               resnet_version: str = '50',  # either 50, 101, or 152
               flatten_output: bool = True,
               circular_pad: bool = False):
    super(ResNetDecoder, self).__init__()

    if flatten_output:
      _check_image_size(image_size)
    self.image_size = image_size
    self.flatten_output = flatten_output
    if self.flatten_output:
      self.upc = tf.keras.Sequential([
          tf.keras.layers.Conv2DTranspose(
              hidden_dims * 2, kernel_size=4, strides=1),
          tf.keras.layers.experimental.SyncBatchNormalization(),
          tf.keras.layers.LeakyReLU(alpha=0.2),
          tf.keras.layers.UpSampling2D(),
      ])
    else:
      self.upc = tf.keras.Sequential([
          tf.keras.layers.Conv2D(
              hidden_dims * 2, kernel_size=1, strides=1, padding='SAME'),
          tf.keras.layers.experimental.SyncBatchNormalization(),
          tf.keras.layers.LeakyReLU(alpha=0.2),
          tf.keras.layers.UpSampling2D(),
      ])

    if resnet_version == '50':
      filters = [6, 4, 3, 3]  # [3, 4, 6, 3]
    elif resnet_version == '101':
      filters = [23, 4, 3, 3]
    elif resnet_version == '152':
      filters = [36, 8, 3, 3]
    else:
      raise ValueError('resnet_version should be one of ["50", "101", "152"], '
                       f'got {resnet_version} instead.')

    if self.flatten_output and self.image_size == 256:
      self.deconv1 = layers.ResStackTranspose(
          hidden_dims * 8,
          hidden_dims * 4,
          filters[0],
          strides=2,
          circular_pad=circular_pad)
    else:
      self.deconv1 = layers.ResStackTranspose(
          hidden_dims * 8,
          hidden_dims * 4,
          filters[0],
          strides=1,
          circular_pad=circular_pad)
    self.deconv2 = layers.ResStackTranspose(
        hidden_dims * 4,
        hidden_dims * 2,
        filters[1],
        strides=2,
        circular_pad=circular_pad)
    self.deconv3 = layers.ResStackTranspose(
        hidden_dims * 2,
        hidden_dims,
        filters[2],
        strides=2,
        circular_pad=circular_pad)
    self.deconv4 = layers.ResStackTranspose(
        hidden_dims,
        hidden_dims,
        filters[3],
        strides=2,
        circular_pad=circular_pad)

    self.agent0 = self.create_agent(hidden_dims)
    self.agent1 = self.create_agent(hidden_dims)
    self.agent2 = self.create_agent(hidden_dims * 2)
    self.agent3 = self.create_agent(hidden_dims * 4)
    self.agent4 = self.create_agent(hidden_dims * 8)

    self.final_conv = layers.ResStackTranspose(
        hidden_dims, hidden_dims, 3, circular_pad=circular_pad)
    self.final_deconv = tf.keras.layers.Conv2DTranspose(
        output_dim, kernel_size=2, strides=2, padding='SAME')

  def call(self, x: tf.Tensor, skip, training=None) -> tf.Tensor:
    out_x = self.upc(x)
    out_x = self.agent4(out_x)  # (8, 8)
    out_x = self.deconv1(out_x)
    out_x = out_x + self.agent3(skip[3])  # (16, 16)
    out_x = self.deconv2(out_x)
    out_x = out_x + self.agent2(skip[2])  # (32, 32)
    out_x = self.deconv3(out_x)
    out_x = out_x + self.agent1(skip[1])  # (64, 64)
    out_x = self.deconv4(out_x)
    out_x = out_x + self.agent0(skip[0])  # (128, 128)
    out_x = self.final_conv(out_x)
    out_x = self.final_deconv(out_x)  # (256, 256)
    return out_x
