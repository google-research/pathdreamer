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

"""Constants."""

CKPT_UNSEEN = 'data/ckpt/structure_gen_ckpt'
SPADE_CKPT_UNSEEN = 'data/ckpt/image_gen_ckpt'

INVALID_RGB_VALUE = -1
INVALID_SEM_VALUE = 0

PI = 3.1415926535897932384626433
HFOV = 90 * PI / 180
DEPTH_SCALE = 10.0

NUM_MP3D_CLASSES = 42

MP3D_ID2CLASS = {
    0: 'void',
    1: 'wall',
    2: 'floor',
    3: 'chair',
    4: 'door',
    5: 'table',
    6: 'picture',
    7: 'cabinet',
    8: 'cushion',
    9: 'window',
    10: 'sofa',
    11: 'bed',
    12: 'curtain',
    13: 'chest_of_drawers',
    14: 'plant',
    15: 'sink',
    16: 'stairs',
    17: 'ceiling',
    18: 'toilet',
    19: 'stool',
    20: 'towel',
    21: 'mirror',
    22: 'tv_monitor',
    23: 'shower',
    24: 'column',
    25: 'bathtub',
    26: 'counter',
    27: 'fireplace',
    28: 'lighting',
    29: 'beam',
    30: 'railing',
    31: 'shelving',
    32: 'blinds',
    33: 'gym_equipment',
    34: 'seating',
    35: 'board_panel',
    36: 'furniture',
    37: 'appliances',
    38: 'clothes',
    39: 'objects',
    40: 'misc',
    41: 'masking',  # Used to handle blurred RGB regions.
}
