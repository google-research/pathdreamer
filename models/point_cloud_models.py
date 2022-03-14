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

"""Models for point cloud processing."""

from typing import List, NamedTuple, Optional, Tuple

from pathdreamer.models import image_models
from pathdreamer.models import layers
import tensorflow as tf


def reparameterize(mu: tf.Tensor, logvar: tf.Tensor) -> tf.Tensor:
  """Reparameterization trick to draw random vector from N(mu, sigma)."""
  sigma = tf.math.exp(0.5 * logvar)
  eps = tf.random.normal(sigma.shape, dtype=mu.dtype)
  return eps * sigma + mu


def compute_kl(mu1: tf.Tensor, logvar1: tf.Tensor, mu2: tf.Tensor,
               logvar2: tf.Tensor) -> tf.Tensor:
  """Computes Kullback-Leibler Divergence for two distributions P and Q.

  Args:
    mu1: Tensor indicating mean value of P. Multi-dimensional tensors are valid.
    logvar1: Tensor indicating log(variance) of P. Expected to have the same
      shape as mu1.
    mu2: Tensor indicating mean value of Q. Expected to have the same shape as
      mu1.
    logvar2: Tensor indicating log(variance) of Q. Expected to have the same
      shape as mu1.

  Raises:
    ValueError: If the argument shapes are not equal.

  Returns:
    kld: Tensor of same shape as mu1 indicating KL divergence between P and Q.
  """
  if not mu1.shape == logvar1.shape == mu2.shape == logvar2.shape:
    raise ValueError('Arguments to compute KLD should have the same shape.')

  sigma1 = tf.math.exp(0.5 * logvar1)
  sigma2 = tf.math.exp(0.5 * logvar2)
  return (tf.math.log(sigma2 / sigma1) + (tf.math.exp(logvar1) +
                                          (mu1 - mu2)**2) /
          (2 * tf.math.exp(logvar2)) - 1 / 2)


class PCOutputData(NamedTuple):
  """Output tuple for point cloud model outputs."""
  pred_logits: tf.Tensor
  pred_depth: tf.Tensor
  kld_loss: tf.Tensor
  mu: tf.Tensor
  logvar: tf.Tensor


class GaussianCNN(tf.keras.Model):
  """Stochastic CNN for predicting latent sequences drawn from a Gaussian."""

  def __init__(self,
               hidden_dims: int = 128,
               output_dim: int = 64,
               kernel_size: int = 3,
               circular_pad: bool = True):
    super(GaussianCNN, self).__init__()
    self.circular_pad = circular_pad

    self.gaussian_cnn = tf.keras.Sequential([
        tf.keras.layers.experimental.SyncBatchNormalization(),
        layers.PadLayer(kernel_size // 2, circular_pad=self.circular_pad),
        tf.keras.layers.Conv2D(
            hidden_dims * 2, kernel_size, strides=1, padding='VALID'),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.experimental.SyncBatchNormalization(),
        tf.keras.layers.Conv2D(
            hidden_dims, kernel_size, strides=1, padding='SAME'),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.experimental.SyncBatchNormalization(),
        tf.keras.layers.Conv2D(
            hidden_dims, kernel_size, strides=1, padding='SAME'),
        tf.keras.layers.LeakyReLU(),
    ])
    self.gaussian_mean = tf.keras.Sequential([
        tf.keras.layers.Conv2D(
            output_dim, kernel_size, strides=1, padding='SAME'),
    ])
    self.gaussian_logvar = tf.keras.Sequential([
        tf.keras.layers.Conv2D(
            output_dim, kernel_size, strides=1, padding='SAME'),
    ])

  def call(self,
           x: tf.Tensor,
           training=None) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Forward pass to sample random vector, mean, and variance.

    Args:
      x: tf.float32 tensor of shape (N, T, H) representing sequence input.
      training: Whether the model is in training model.

    Returns:
      z: tf.float32 tensor of shape (N, output_dim) of random noise. Drawn from
        a Gaussian with mean = mu and variance = exp(logvar).
      mu: tf.float32 tensor of shape (N, output_dim) of mean of the Gaussian.
      logvar: tf.float32 tensor of shape (N, output_dim) describing log(sigma^2)
        of the Gaussian.
    """
    out = self.gaussian_cnn(x, training=training)  # (N, hidden_dims)
    mu = self.gaussian_mean(out, training=training)
    logvar = self.gaussian_logvar(out, training=training)
    z = reparameterize(mu, logvar)
    return z, mu, logvar


class ResNetSeqInpainter(tf.keras.Model):
  """U-Net style architecture for inpainting RGB / depth sequence data."""

  def __init__(self,
               image_height: int = 256,
               hidden_dims: int = 64,
               resnet_version: str = '50',
               z_dim: int = 64,
               flatten: bool = True,
               circular_pad: bool = True,
               random_noise: bool = False,
               noise_mixup_gt_prob: float = 1.0):
    """Initializes an inpainting model.

    Args:
      image_height: Image size to train model on.
      hidden_dims: Base dimensions of hidden units.
      resnet_version: ResNet version to use for image models. Should be one of
        ['50', '101', '152'].
      z_dim: Dimension of noise vector.
      flatten: If false, encoder and decoder are fully convolutional networks.
      circular_pad: If true, applies circular padding.
      random_noise: If true, learns a random noise embedding similar to the
        Stochastic Video Generation model.
      noise_mixup_gt_prob: Probability of using the groundtruth noise embedding
        during learning. A value of 1.0 indicates the groundtruth is always
        used. During inference, the noise is sampled from the learnt prior.
    """
    super(ResNetSeqInpainter, self).__init__()
    self.image_height = image_height
    self.hidden_dims = hidden_dims
    self.resnet_version = resnet_version
    self.flatten = flatten
    self.circular_pad = circular_pad
    self.random_noise = random_noise
    self.noise_mixup_gt_prob = noise_mixup_gt_prob
    self.z_dim = z_dim

  def build(self, input_shapes):
    # Output will be same dimensionality as input.
    total_dims = sum([input_shapes[i][-1] for i in range(len(input_shapes))])

    self.encoder = image_models.ResNetEncoder(
        image_size=self.image_height,
        hidden_dims=self.hidden_dims,
        resnet_version=self.resnet_version,
        flatten_output=self.flatten,
        circular_pad=self.circular_pad)
    self.decoder = image_models.ResNetDecoder(
        output_dim=self.hidden_dims,
        image_size=self.image_height,
        hidden_dims=self.hidden_dims,
        resnet_version=self.resnet_version,
        flatten_output=self.flatten,
        circular_pad=self.circular_pad)
    self.shortcut_conv = tf.keras.Sequential([
        tf.keras.layers.experimental.SyncBatchNormalization(),
        layers.PadLayer(1, circular_pad=self.circular_pad),
        tf.keras.layers.Conv2D(
            self.hidden_dims, kernel_size=3, strides=1, padding='VALID'),
    ])
    self.gate_conv = tf.keras.Sequential([
        tf.keras.layers.experimental.SyncBatchNormalization(),
        layers.PadLayer(1, circular_pad=self.circular_pad),
        tf.keras.layers.Conv2D(
            self.hidden_dims, kernel_size=3, strides=1, padding='VALID'),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.experimental.SyncBatchNormalization(),
        layers.PadLayer(1, circular_pad=self.circular_pad),
        tf.keras.layers.Conv2D(1, kernel_size=3, strides=1, padding='VALID'),
        tf.keras.layers.Activation('sigmoid'),
    ])
    self.final_conv = tf.keras.Sequential([
        layers.PadLayer(1, circular_pad=self.circular_pad),
        tf.keras.layers.Conv2D(
            total_dims, kernel_size=3, strides=1, padding='VALID'),
    ])

    if self.random_noise:
      self.prior = GaussianCNN(
          hidden_dims=self.hidden_dims,
          circular_pad=self.circular_pad,
          output_dim=self.z_dim)
      self.posterior = GaussianCNN(
          hidden_dims=self.hidden_dims,
          circular_pad=self.circular_pad,
          output_dim=self.z_dim)

  def call(self,
           inputs: List[tf.Tensor],
           groundtruth_inputs: Optional[List[tf.Tensor]] = None,
           sample_noise: bool = True,
           z: Optional[tf.Tensor] = None,
           training=None) -> PCOutputData:
    """Performs inpainting over an RGB and depth tensor.

    Args:
      inputs: List of 2 tf.Tensors (RGB and D maps) to perform inpainting on.
        The RGB tensor has shape (N, T, H, W, C) and the depth tensor has shape
        (N, T, H, W, 1).
      groundtruth_inputs: Groundtruth tensors.
      sample_noise: If True, samples noise from the prior distribution. If this
        is False, it uses mu directly.
      z: Random noise to use. If None, samples from the conditional prior.
      training: Whether the model is in training mode.
    Returns:
      outputs: Output tuple of tf.Tensor values with same shapes as the inputs.
    Raises:
      ValueError: If inputs contains an invalid number of items.
    """
    if len(inputs) != 2:
      raise ValueError('Input should contain 2 items (RGB and depth tensors).')

    pred_feat_tensor, pred_depth_tensor = inputs
    batch_size, pred_len, height, width, _ = pred_depth_tensor.shape
    # Merge batch and time dimensions.
    pred_depth_tensor_merged = tf.reshape(
        pred_depth_tensor, (batch_size * pred_len, height, width, -1))

    # Run forward pass.
    # Merge batch and time dimensions.
    pred_feat_tensor_merged = tf.reshape(
        pred_feat_tensor, (batch_size * pred_len, height, width, -1))
    input_dims = [
        pred_feat_tensor_merged.shape[-1], pred_depth_tensor_merged.shape[-1]]
    combined_input = tf.concat(
        [pred_feat_tensor_merged, pred_depth_tensor_merged], axis=-1)

    hidden_spatial, skip = self.encoder(combined_input)
    if self.flatten:
      # Convert hidden to a (N, 1, 1, C) tensor for decoder.
      hidden_spatial = hidden_spatial[:, None, None, :]
    kld_loss = tf.zeros(hidden_spatial.shape[:-1] + (self.z_dim,))
    mu_p = tf.zeros(hidden_spatial.shape[:-1] + (self.z_dim,))
    logvar_p = tf.zeros(hidden_spatial.shape[:-1] + (self.z_dim,))

    # Add conditional random noise.
    if self.random_noise:
      if z is not None:
        hidden_spatial = tf.concat([hidden_spatial, z], axis=-1)
      else:
        z, mu_p, logvar_p = self.prior(hidden_spatial)
        if training:
          # Use groundtruth tensor for training prior.
          gt_feat_tensor, gt_depth_tensor = groundtruth_inputs
          # Merge batch and time dimensions.
          gt_depth_tensor_merged = tf.reshape(
              gt_depth_tensor, (batch_size * pred_len, height, width, -1))
          # Merge batch and time dimensions.
          gt_feat_tensor_merged = tf.reshape(
              gt_feat_tensor, (batch_size * pred_len, height, width, -1))
          gt_combined_input = tf.concat(
              [gt_feat_tensor_merged, gt_depth_tensor_merged], axis=-1)
          gt_hidden_spatial, _ = self.encoder(gt_combined_input, training=False)
          z_t, mu, logvar = self.posterior(
              tf.stop_gradient(gt_hidden_spatial), training=training)
          kld_loss = compute_kl(mu, logvar, mu_p, logvar_p)

          # Randomly sample from posterior / prior.
          if tf.random.uniform((), 0, 1) < self.noise_mixup_gt_prob:
            z = z_t
          hidden_spatial = tf.concat([hidden_spatial, z], axis=-1)
        else:
          # Evaluation, sample from prior or use mean.
          if sample_noise:
            hidden_spatial = tf.concat([hidden_spatial, z], axis=-1)
          else:
            hidden_spatial = tf.concat([hidden_spatial, mu_p], axis=-1)

    out = self.decoder(hidden_spatial, skip)

    # Shortcut from input to output.
    gating = self.gate_conv(combined_input)
    out = gating * self.shortcut_conv(combined_input) + (1 - gating) * out
    out = self.final_conv(out)

    feat, depth = tf.split(out, input_dims, axis=-1)
    # Reshape back to sequence format.
    pred_logits = tf.reshape(feat, (batch_size, pred_len, height, width, -1))

    depth = tf.math.sigmoid(depth)

    # Reshape back to sequence format.
    pred_depth = tf.reshape(depth, (batch_size, pred_len, height, width, 1))
    kld_loss = tf.reshape(kld_loss, (batch_size, pred_len, kld_loss.shape[1],
                                     kld_loss.shape[2], kld_loss.shape[3]))
    mu_p = tf.reshape(mu_p, (batch_size, pred_len, mu_p.shape[1], mu_p.shape[2],
                             mu_p.shape[3]))
    logvar_p = tf.reshape(logvar_p, (batch_size, pred_len, logvar_p.shape[1],
                                     logvar_p.shape[2], logvar_p.shape[3]))
    return PCOutputData(
        pred_logits=tf.cast(pred_logits, tf.float32),
        pred_depth=tf.cast(pred_depth, tf.float32),
        kld_loss=tf.cast(kld_loss, tf.float32),
        mu=tf.cast(mu_p, tf.float32),
        logvar=tf.cast(logvar_p, tf.float32))
