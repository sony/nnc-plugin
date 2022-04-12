# Copyright 2021 Sony Group Corporation.
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


def is_normalized(im):
    return im.max() <= 1.0


def not_normalized(im):
    return not is_normalized(im)


def preprocess_image(image, no_image_normalization):
    im = image.copy()
    # Single-image version input is not normalized whatever no_image_normalization is
    # Since the input is not generated through data_iterator
    if (not no_image_normalization) & (not_normalized(im)):
        im = im / 255.0
    if len(im.shape) < 3:
        im = im.reshape((1,) + im.shape)
    return im


def overlay_images(im1, im2, im1_ratio=0.5):
    return (im1 * im1_ratio + im2 * (1.0 - im1_ratio)).astype(np.uint8)


def add_frame(im, output_shape, pad):
    start_h, end_h = pad[0], output_shape[0] - pad[0]
    start_w, end_w = pad[1], output_shape[1] - pad[1]
    _h, _w = output_shape
    ret = imresize(np.zeros((3, 1, 1), np.uint8),
                   (_w, _h), channel_first=True).copy()
    ret[:, start_h:end_h, start_w:end_w] = im
    return ret


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


def get_gray_image(img, percentile=99):
    ret = np.sum(np.abs(img), axis=0)
    vmax = np.percentile(ret, percentile)
    vmin = np.percentile(ret, 100 - percentile)
    ret = np.clip((ret - vmin) / (vmax - vmin), 0, 1)
    return (ret * 255).astype(np.uint8)


def normalize_image(layer):
    ret = layer.mean(axis=(0, 1))
    max_v, min_v = np.max(ret), np.min(ret)
    if max_v != min_v:
        ret = (ret - min_v) / (max_v - min_v)
    return ret
