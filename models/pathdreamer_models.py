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

"""Interface for using pretrained Pathdreamer models for prediction."""

from typing import NamedTuple, Optional

from pathdreamer import constants
from pathdreamer.models import pathdreamer_config
from pathdreamer.models import point_cloud_models
from pathdreamer.models import spade_models
from pathdreamer.utils import pano_utils
import tensorflow as tf


class PanoData(NamedTuple):
  """Data corresponding to a Matterport3D panorama.

  position: (3,) float tensor of xyz coordinates.
  rgb: (H, W, 3) int32 tensor of RGB panorama image.
  semantic: (H, W) int32 tensor of semantic segmentation panorama image.
  depth: (H, W) float32 tensor of depth results, with values ranging [0, 1].
  """
  position: tf.Tensor
  rgb: tf.Tensor
  semantic: tf.Tensor
  depth: tf.Tensor


class PathdreamerOutputData(NamedTuple):
  """Output tuple for Pathdreamer model outputs.

  proj_semantic: (N, H, W) int tensor with values ranging [0, num_classes].
  pred_semantic: (N, H, W) int tensor with values ranging [0, num_classes].
  proj_rgb: (N, H, W, 3) int tensor with values ranging [0, 255].
  pred_rgb: (N, H, W, 3) int tensor with values ranging [0, 255].
  proj_depth: (N, H, W) tensor with values ranging [0, 1].
  pred_depth: (N, H, W) tensor with values ranging [0, 1].
  mu: (N, H', W', Z) tensor.
  logvar: (N, H', W', Z) tensor.
  """
  proj_semantic: tf.Tensor
  pred_semantic: tf.Tensor
  proj_rgb: tf.Tensor
  pred_rgb: tf.Tensor
  proj_depth: tf.Tensor
  pred_depth: tf.Tensor
  mu: tf.Tensor
  logvar: tf.Tensor


class PathdreamerMemoryState(NamedTuple):
  """Tuple for Pathdreamer memory state.

  coords: (N, 4, M) float tensor containing (x, y, z, 1) coordinates.
  feats: (N, M) int tensor containing semantic features.
  rgb: (N, M, 3) int tensor containing RGB features in the [0, 255] range.
  """
  coords: tf.Tensor
  feats: tf.Tensor
  rgb_coords: tf.Tensor
  rgb: tf.Tensor


class PathdreamerModel(object):
  """Interface to use a pretrained Pathdreamer model for predictions."""

  def __init__(self,
               config: pathdreamer_config.PathdreamerConfig = pathdreamer_config
               .get_config()):
    self.config = config
    if config.batch_size != 1:
      print('Several methods do not support batch_size > 1.')
    self.model = point_cloud_models.ResNetSeqInpainter(
        hidden_dims=config.hidden_dims,
        flatten=False,
        circular_pad=config.circular_pad,
        random_noise=config.random_noise,
        z_dim=config.z_dim)
    if config.ckpt_path is not None:
      ckpt_path = config.ckpt_path
      ckpt = tf.train.Checkpoint(model=self.model)
      status = ckpt.restore(ckpt_path)
      status.assert_existing_objects_matched()
      print(f'Restored from {ckpt_path}')
    else:
      print('Initializing Pathdreamer model from scratch.')

    self.spade_model = spade_models.WCSPADEGenerator(
        batch_size=config.batch_size,
        gen_dims=config.spade_gen_dims,
        z_dim=config.spade_z_dim,
        image_size=config.image_height,
        fully_conv=True,
        circular_pad=config.circular_pad,
        use_depth_condition=True)
    if config.spade_ckpt_path is not None:
      spade_ckpt_path = config.spade_ckpt_path
      spade_ckpt = tf.train.Checkpoint(ema_generator=self.spade_model)
      status = spade_ckpt.restore(spade_ckpt_path)
      status.assert_existing_objects_matched()
      print(f'Restored SPADE from {spade_ckpt_path}')
    else:
      print('Initializing SPADE model from scratch.')

    self.prev_rgb_frame = None
    self.batch_size = config.batch_size
    self.height = config.image_height
    self.width = config.image_height * 2
    self.pathdreamer_height = int(self.height *
                                  config.pathdreamer_height_multiplier)
    self.pathdreamer_width = int(self.width *
                                 config.pathdreamer_height_multiplier)
    self.pathdreamer_size_mult = config.pathdreamer_height_multiplier
    self.reset_memory()

  def _check_batch_size(self, input_batch_size):
    if input_batch_size != self.batch_size:
      raise ValueError('Input batch size is not suitable. Expected '
                       f'{self.batch_size}, got {input_batch_size} instead.')

  def _transform_position(self, xyz):
    """Transforms xyz coords into the expected internal representation."""
    transformed = tf.stack(
        [xyz[:, 0], xyz[:, 1], xyz[:, 2],
         tf.zeros(self.batch_size,)], axis=1)
    return transformed

  def reset_memory(self):
    """Resets memory to zeros."""
    self._memory = PathdreamerMemoryState(
        coords=tf.zeros((self.batch_size, 4, 0)),
        feats=tf.zeros((self.batch_size, 0, 1), dtype=tf.uint8),
        rgb_coords=tf.zeros((self.batch_size, 4, 0)),
        rgb=tf.zeros((self.batch_size, 0, 3), dtype=tf.int32),
    )

  def get_memory_state(self) -> PathdreamerMemoryState:
    """Returns a copy of the current memory."""
    return PathdreamerMemoryState(
        coords=tf.identity(self._memory.coords),
        feats=tf.identity(self._memory.feats),
        rgb_coords=tf.identity(self._memory.rgb_coords),
        rgb=tf.identity(self._memory.rgb),
    )

  def set_memory_state(self, state: PathdreamerMemoryState):
    """Sets memory to the given state."""
    self._memory = PathdreamerMemoryState(
        coords=tf.identity(state.coords),
        feats=tf.identity(state.feats),
        rgb_coords=tf.identity(state.rgb_coords),
        rgb=tf.identity(state.rgb),
    )

  def write_memory_as_pointcloud(self, filename):
    """Writes memory at batch position 0 to .ply text file."""
    memory_state = self.get_memory_state()
    xyz_pts = memory_state.rgb_coords[0, 0:3].numpy().T
    rgb_pts = memory_state.rgb[0].numpy()

    # Write header of .ply file
    fp = tf.io.gfile.GFile(filename, 'w')
    fp.write('ply\n')
    fp.write('format ascii 1.0 \n')
    fp.write('element vertex %d\n' % xyz_pts.shape[0])
    fp.write('property float x\n')
    fp.write('property float y\n')
    fp.write('property float z\n')
    fp.write('property uchar red\n')
    fp.write('property uchar green\n')
    fp.write('property uchar blue\n')
    fp.write('end_header\n')

    # Write 3D points to .ply file
    for i in range(xyz_pts.shape[0]):
      fp.write('{} {} {} {} {} {} \n'.format(xyz_pts[i, 0], xyz_pts[i, 1],
                                             xyz_pts[i, 2], rgb_pts[i, 0],
                                             rgb_pts[i, 1], rgb_pts[i, 2]))
    fp.close()

  def add_to_memory(self,
                    pano_rgb,
                    pano_semantic,
                    pano_depth,
                    position,
                    mask_blurred=True):
    """Add an equirectangular observation to the memory.

    Args:
      pano_rgb: (N, H, W, 3) equirectangular RGB image with values in [0, 255].
      pano_semantic: (N, H, W, 1) equirectangular image of segmentation classes.
        The heading of this image is expected to be in the R2R orientation.
      pano_depth: (N, H, W) equirectangular depth image with values in [0, 1].
        The heading of this image is expected to be in the R2R orientation.
      position: (N, 3) of xyz position coordinates. The y-axis points to the
        center of the pano image, the z-axis points to the north pole, and the
        x-axis points to the horizon between the center and the right hand side.
        This is equivalent to the R2R dataset format, i.e. when loading from
        Matterport connectivity data use (pose[3], pose[7], pose[11]).
      mask_blurred: If True, asks the top and bottom 1/8th of the image.

    Returns:
      Nothing. Memory coords and feats are updated with the new data.
    """
    self._check_batch_size(pano_semantic.shape[0])
    assert pano_rgb.dtype in [tf.uint8, tf.int32]
    assert pano_semantic.dtype in [tf.uint8, tf.int32]
    pano_rgb = tf.cast(pano_rgb, tf.int32)
    pano_semantic = tf.cast(pano_semantic, tf.uint8)

    self.prev_rgb_frame = tf.cast(pano_rgb / 255, tf.float32)
    if mask_blurred:
      pano_rgb = pano_utils.mask_pano(pano_rgb)

    # If this is the first item, use this as the origin.
    transformed_position = self._transform_position(position)
    xyz1, feats = pano_utils.equirectangular_to_pointcloud(
        pano_semantic, pano_depth, constants.INVALID_SEM_VALUE,
        self.config.depth_scale,
        self.pathdreamer_size_mult * self.config.size_mult)
    rgb_xyz1, rgb_feats = pano_utils.equirectangular_to_pointcloud(
        pano_rgb, pano_depth, constants.INVALID_RGB_VALUE,
        self.config.depth_scale, self.config.size_mult)

    # Offset by appropriate coordinates.
    xyz1 += transformed_position[:, :, None]
    rgb_xyz1 += transformed_position[:, :, None]

    # Filter coords if they are not valid.
    feats_valid = tf.reduce_any(
        feats != constants.INVALID_SEM_VALUE, axis=(0, 2))
    filtered_xyz1 = tf.gather(xyz1, tf.where(feats_valid), axis=2)[..., 0]
    filtered_feats = tf.gather(feats, tf.where(feats_valid), axis=1)[..., 0]
    rgb_valid = tf.reduce_any(
        rgb_feats != constants.INVALID_RGB_VALUE, axis=(0, 2))
    filtered_rgb_xyz1 = tf.gather(rgb_xyz1, tf.where(rgb_valid), axis=2)[..., 0]
    filtered_rgb = tf.gather(rgb_feats, tf.where(rgb_valid), axis=1)[..., 0, :]
    filtered_rgb = tf.cast(filtered_rgb, self._memory.rgb.dtype)

    new_memory_state = PathdreamerMemoryState(
        coords=tf.concat([self._memory.coords, filtered_xyz1], axis=2),
        feats=tf.concat([self._memory.feats, filtered_feats], axis=1),
        rgb_coords=tf.concat(
            [self._memory.rgb_coords, filtered_rgb_xyz1], axis=2),
        rgb=tf.concat([self._memory.rgb, filtered_rgb], axis=1))
    self.set_memory_state(new_memory_state)

  def __call__(self,
               position: tf.Tensor,
               add_preds_to_memory: bool = False,
               sample_noise: bool = True,
               use_projected_pathdreamer_rgb: bool = False,
               z: Optional[tf.Tensor] = None) -> PathdreamerOutputData:
    """Runs forward pass to predict the frame at a given position.

    Args:
      position: (N, 3) tensor describing coordinates to create a prediction at.
        These are the xyz (i.e. (pose[3], pose[7], pose[11]) points) from R2R.
      add_preds_to_memory: If True, adds predicted coords and features back into
        memory. This will be used for any following predictions.
      sample_noise: If True, samples noise from the prior distribution. If this
        is False, it uses mu directly.
      use_projected_pathdreamer_rgb: If False, returns Pathdreamer RGB outputs
        directly. If True, any projected RGB pixels will be forced to remain as
        their original value. This may be set to True for more consistent
        generation of continuous videos.
      z: Random noise to use. If None, samples from the conditional prior.

    Returns:
      output: PathdreamerOutputData named tuple containing predictions.
    """
    batch_size = position.shape[0]
    self._check_batch_size(batch_size)
    # Offset memory by appropriate coordinates.
    relative_position = self._transform_position(position)
    relative_coords = self._memory.coords - relative_position[..., None]
    relative_rgb_coords = self._memory.rgb_coords - relative_position[..., None]
    proj_depth, proj_semantic = pano_utils.project_feats_to_equirectangular(
        self._memory.feats, relative_coords, self.pathdreamer_height,
        self.pathdreamer_width, constants.INVALID_SEM_VALUE,
        self.config.depth_scale)
    _, proj_rgb = pano_utils.project_feats_to_equirectangular(
        self._memory.rgb, relative_rgb_coords, self.height, self.width,
        constants.INVALID_RGB_VALUE, self.config.depth_scale)
    # Remove channels dim of 1.
    proj_semantic = proj_semantic[..., 0]
    proj_semantic = tf.cast(proj_semantic, tf.uint8)
    proj_rgb = tf.cast(proj_rgb, tf.uint8)
    one_hot_feat = tf.one_hot(proj_semantic, constants.NUM_MP3D_CLASSES)
    # Run predictions through inpainting model.
    # Rescale inputs to match Pathdreamer proportions.
    proj_depth = tf.image.resize(
        proj_depth[..., None],
        (self.pathdreamer_height, self.pathdreamer_width), 'nearest')
    pred_logits, pred_depth, _, mu, logvar = self.model(
        [one_hot_feat[:, None, ...], proj_depth[:, None, ...]],
        sample_noise=sample_noise,
        z=z,
        training=False)
    mu = mu[:, 0, ...]
    logvar = logvar[:, 0, ...]
    pred_depth = tf.image.resize(
        pred_depth[:, 0, ...], (self.height, self.width), 'nearest')[..., 0]
    pred_semantic = tf.argmax(
        pred_logits[:, 0, ...], axis=-1, output_type=tf.int32)
    pred_semantic = tf.cast(pred_semantic, tf.uint8)

    pred_semantic = tf.image.resize(
        pred_semantic[..., None], (self.height, self.width), 'nearest')[..., 0]
    proj_semantic = tf.image.resize(
        proj_semantic[..., None], (self.height, self.width), 'nearest')[..., 0]
    proj_depth = tf.image.resize(
        proj_depth, (self.height, self.width), 'nearest')[..., 0]
    proj_mask = tf.cast(
        tf.reduce_all(proj_rgb != 0, axis=-1), tf.float32)[..., None]

    assert self.prev_rgb_frame is not None
    spade_cond = {
        'one_hot_mask':
            tf.one_hot(pred_semantic,
                       constants.NUM_MP3D_CLASSES),  # (N, H, W, 41)
        'prev_image':
            self.prev_rgb_frame,  # (N, H, W, 3)
        'proj_image':
            tf.cast(proj_rgb / 255, tf.float32),  # (N, H, W, 3)
        'proj_mask':
            proj_mask,  # (N, H, W, 1)
        'depth':
            pred_depth[..., None],  # (N, H, W, 1)
        'first_frame':
            tf.zeros((batch_size,)),  # (N,)
    }

    _, _, generated_pred_rgb = self.spade_model([spade_cond, None])
    pred_rgb = tf.cast(generated_pred_rgb * 255, tf.uint8)

    if add_preds_to_memory:
      # Mask out features that came from the point cloud.
      pred_rgb_mem = tf.cast(1 - proj_mask, pred_rgb.dtype) * pred_rgb
      pred_semantic_mem = tf.cast(1 - proj_mask[..., 0],
                                  pred_semantic.dtype) * pred_semantic
      pred_depth_mem = tf.cast(1 - proj_mask[..., 0],
                               pred_depth.dtype) * pred_depth + (tf.cast(
                                   proj_mask[..., 0], pred_depth.dtype)) * 1.0
      if use_projected_pathdreamer_rgb:
        pred_rgb = proj_rgb + pred_rgb_mem
        pred_semantic = proj_semantic + pred_semantic_mem
        pred_depth = proj_depth + pred_depth_mem
        generated_pred_rgb = tf.cast(pred_rgb_mem / 255, tf.float32)
      self.prev_rgb_frame = generated_pred_rgb
      self.add_to_memory(
          pred_rgb_mem,
          pred_semantic_mem[..., None],
          pred_depth_mem,
          position)

    return PathdreamerOutputData(
        proj_semantic=proj_semantic,
        pred_semantic=pred_semantic,
        proj_rgb=proj_rgb,
        pred_rgb=pred_rgb,
        proj_depth=proj_depth,
        pred_depth=pred_depth,
        mu=mu,
        logvar=logvar)
