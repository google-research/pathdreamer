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

"""Config object for Pathdreamer models."""

from typing import List, Optional

from pathdreamer import constants


class PathdreamerConfig:
  """Parameters used to configure Pathdreamer models."""
  batch_size: int = 1
  ckpt_path: Optional[str] = constants.CKPT_UNSEEN
  spade_ckpt_path: Optional[str] = constants.SPADE_CKPT_UNSEEN
  hidden_dims: int = 128
  random_noise: bool = True
  z_dim: int = 32
  circular_pad: bool = True
  spade_gen_dims: int = 96
  spade_z_dim: int = 256
  size_mult: int = 1
  image_height: int = 512
  pathdreamer_height_multiplier: float = 0.5
  depth_scale: float = constants.DEPTH_SCALE
  scan_ids: Optional[List[str]] = None
  h_fov: float = 0.17


def get_config() -> PathdreamerConfig:
  config = PathdreamerConfig()
  return config


def get_test_config() -> PathdreamerConfig:
  """Returns config used for unit tests."""
  config = PathdreamerConfig()
  config.ckpt_path = None
  config.spade_ckpt_path = None
  config.hidden_dims = 4
  config.z_dim = 4
  config.spade_gen_dims = 4
  config.spade_z_dim = 4
  return config
