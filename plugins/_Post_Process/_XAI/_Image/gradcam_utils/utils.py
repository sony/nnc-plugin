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
import csv
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


def add_frame(im, output_shape, pad):
    start_h, end_h = pad[0], output_shape[0] - pad[0]
    start_w, end_w = pad[1], output_shape[1] - pad[1]
    _h, _w = output_shape
    ret = imresize(np.zeros((3, 1, 1), np.uint8),
                   (_w, _h), channel_first=True).copy()
    ret[:, start_h:end_h, start_w:end_w] = im
    return ret


def overlay_images(im1, im2, im1_ratio=0.5):
    return (im1 * im1_ratio + im2 * (1.0 - im1_ratio)).astype(np.uint8)


def save_info_to_csv(input_path, output_path, file_names):
    with open(input_path, newline='') as f:
        rows = [row for row in csv.reader(f)]
    row0 = rows.pop(0)
    row0.append('gradcam')
    for i, file_name in enumerate(file_names):
        rows[i].append(file_name)
    with open(output_path, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(row0)
        writer.writerows(rows)


def get_num_classes(input_path, label_variable):
    csv_file = np.loadtxt(input_path, delimiter=",", dtype=str)
    header = [item.split(":")[0] for item in csv_file[0]]
    classes = csv_file[1:, header.index(label_variable)]
    num_classes = np.unique(classes).size
    return num_classes


def is_binary_classification(num_classes, output_classes):
    return num_classes != output_classes


def get_class_index(label, is_binary_clsf):
    if is_binary_clsf:
        class_index = 0
    else:
        class_index = np.argmax(label)
    return int(class_index)
