# Copyright 2023 Sony Group Corporation.
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


def has_specific_layer(variables, layer_name):
    for k in variables.keys():
        if layer_name in k:
            return True
    return False


def get_last_conv_name(variables, layer_name="Convolution"):
    for k in reversed(variables.keys()):
        if layer_name in k:
            return k
    return None


def get_first_conv_name(variables, layer_name="Convolution"):
    for k in variables.keys():
        if layer_name in k:
            return k
    return None


def get_layer_shape(variables, layer_name):
    return variables[layer_name].shape


def get_layer_name_from_idx(variables, idx):
    if (idx < 0) | (len(variables.keys()) <= idx):
        msg = "Layer index must be in the followings.\n"
        for i, k in enumerate(variables.keys()):
            msg += f'{i}: {k}\n'
        raise IndexError(msg)
    return list(variables.keys())[idx]
