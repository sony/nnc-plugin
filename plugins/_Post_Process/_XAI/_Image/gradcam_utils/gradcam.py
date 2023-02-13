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
import numpy as np
from nnabla.utils.image_utils import imresize
from nnabla.utils.cli.utility import let_data_to_variable
from .utils import add_frame, overlay_images, preprocess_image, is_normalized


def value_to_color(value):
    if value < 1.0 / 8:
        value += 1.0 / 8
        return [0, 0, value * 255 * 4]
    elif value < 3.0 / 8:
        value -= 1.0 / 8
        return [0, value * 255 * 4, 255]
    elif value < 5.0 / 8:
        value -= 3.0 / 8
        return [value * 255 * 4, 255, 255 - value * 255 * 4]
    elif value < 7.0 / 8:
        value -= 5.0 / 8
        return [255, 255 - value * 255 * 4, 0]
    else:
        value -= 7.0 / 8
        return [255 - value * 255 * 4, 0, 0]


def generate_heatmap(im):
    hm_sh = im.shape
    ret = np.ndarray((3,) + hm_sh)
    for y in range(hm_sh[0]):
        for x in range(hm_sh[1]):
            ret[:, y, x] = value_to_color(im[y, x])
    return ret


def gradcam(conv_output, conv_grad):
    pooled_grad = conv_grad.mean(axis=(0, 2, 3), keepdims=True)
    ret = pooled_grad * conv_output
    ret = np.maximum(ret, 0)    # ReLU
    ret = ret.mean(axis=(0, 1))
    max_v, min_v = np.max(ret), np.min(ret)
    if max_v != min_v:
        ret = (ret - min_v) / (max_v - min_v)
    return ret


def zero_grad(executor):
    for k, v in executor.network.variables.items():
        v.variable_instance.need_grad = True
        v.variable_instance.grad.zero()


def get_gradcam_image(input_image, config):
    executor = config.executor
    output_variable = config.output_variable
    input_variable = config.input_variable
    # input image
    im = preprocess_image(input_image, executor.no_image_normalization)
    let_data_to_variable(
        input_variable.variable_instance,
        im,
        data_name=config.input_data_name,
        variable_name=input_variable.name
    )
    # forward, backward
    zero_grad(executor)
    selected = output_variable.variable_instance[:, config.class_index]
    selected.forward()
    selected.backward()

    # Grad-CAM
    last_conv = executor.network.variables[config.last_conv_name]
    conv_grad = last_conv.variable_instance.g.copy()
    conv_output = last_conv.variable_instance.d.copy()
    gc = gradcam(conv_output, conv_grad)

    # Generate output image
    heatmap = generate_heatmap(gc)
    _h, _w = config.cropped_shape
    heatmap = imresize(heatmap, (_w, _h), channel_first=True)
    heatmap = add_frame(heatmap, config.input_shape, config.padding)
    _im = input_image.copy()
    if is_normalized(_im):
        _im = _im * 255.0
    result = overlay_images(heatmap, _im)
    return result
