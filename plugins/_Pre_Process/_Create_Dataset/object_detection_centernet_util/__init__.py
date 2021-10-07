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
import copy
import csv
import math
import multiprocessing as mp
import os
import random
import re

import nnabla.logger as logger
import numpy as np
import tqdm
from nnabla.utils.image_utils import imread, imresize, imsave


class ObjectRect:
    def __init__(self, LRTB=None, XYWH=None):
        if LRTB is not None:
            self.rect = np.array(LRTB)
        elif XYWH is not None:
            self.rect = np.array([XYWH[0] - XYWH[2] * 0.5, XYWH[1] - XYWH[3]
                                  * 0.5, XYWH[0] + XYWH[2] * 0.5, XYWH[1] + XYWH[3] * 0.5])
        else:
            self.rect = np.full((4,), 0.0, dtype=np.float)

    def clip(self):
        return ObjectRect(LRTB=self.rect.clip(0.0, 1.0))

    def left(self):
        return self.rect[0]

    def top(self):
        return self.rect[1]

    def right(self):
        return self.rect[2]

    def bottom(self):
        return self.rect[3]

    def width(self):
        return np.max(self.rect[2] - self.rect[0], 0)

    def height(self):
        return np.max(self.rect[3] - self.rect[1], 0)

    def centerx(self):
        return (self.rect[0] + self.rect[2]) * 0.5

    def centery(self):
        return (self.rect[1] + self.rect[3]) * 0.5

    def center(self):
        return self.centerx(), self.centery()

    def area(self):
        return self.width() * self.height()

    def overlap(self, rect2):
        w = np.max([np.min([self.right(), rect2.right()]) -
                    np.max([self.left(), rect2.left()])], 0)
        h = np.max([np.min([self.bottom(), rect2.bottom()]) -
                    np.max([self.top(), rect2.top()])], 0)
        return w * h

    def iou(self, rect2):
        overlap = self.overlap(rect2)
        return overlap / (self.area() + rect2.area() - overlap)


def load_label(file_name):
    labels = []
    if os.path.exists(file_name):
        with open(file_name, "rt") as f:
            lines = f.readlines()
        for line in lines:
            values = [float(s) for s in line.split(' ')]
            if len(values) == 5:
                labels.append(values)
    else:
        logger.warning(
            "Label txt file is not found %s." % (file_name))
    return labels


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 - sq1) / (2 * a1)

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 - sq2) / (2 * a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / (2 * a3)

    return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius +
                               bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def trim_warp(label, input_size, output_size):
    w_scale = input_size[0] * 1.0 / output_size[0]
    h_scale = input_size[1] * 1.0 / output_size[1]
    label[1] = (label[1] - (1.0 - 1.0 / w_scale)
                * 0.5) * w_scale
    label[2] = (label[2] - (1.0 - 1.0 / h_scale)
                * 0.5) * h_scale
    label[3] *= w_scale
    label[4] *= h_scale
    return label


def pad_warp(label, input_size, output_size):
    w_scale = input_size[0] * 1.0 / output_size[0]
    h_scale = input_size[1] * 1.0 / output_size[1]
    label[1] = (label[1] * w_scale + (1.0 - w_scale) * 0.5)
    label[2] = (label[2] * h_scale + (1.0 - h_scale) * 0.5)
    label[3] *= w_scale
    label[4] *= h_scale
    return label


def create_file_list(source_dir, dir=""):
    result = []
    items = os.listdir(os.path.join(source_dir, dir))
    for item in items:
        if os.path.isdir(os.path.join(source_dir, dir, item)):
            result.extend(create_file_list(
                source_dir=source_dir, dir=os.path.join(dir, item)))
        elif re.search(r'\.(bmp|jpg|jpeg|png|gif|tif|tiff)', os.path.splitext(item)[1], re.IGNORECASE):
            result.append(os.path.join(dir, item))
    return result


def convert_image(process_dict):
    file_name = process_dict['data']
    source_dir = process_dict['source_dir']
    dest_dir = process_dict['dest_dir']
    width = process_dict['width']
    height = process_dict['height']
    mode = process_dict['mode']
    ch = process_dict['ch']
    num_class = process_dict['num_class']
    grid_size = process_dict['grid_size']

    src_file_name = os.path.join(source_dir, file_name)
    src_label_file_name = os.path.join(
        source_dir, os.path.splitext(file_name)[0] + ".txt")
    image_file_name = os.path.join(
        dest_dir, 'data', os.path.splitext(file_name)[0] + ".png")
    label_file_name = os.path.join(
        dest_dir, 'data', os.path.splitext(file_name)[0] + "_label.csv")
    region_file_name = os.path.join(
        dest_dir, 'data', os.path.splitext(file_name)[0] + "_region.csv")
    try:
        os.makedirs(os.path.dirname(image_file_name))
    except OSError:
        pass  # python2 does not support exists_ok arg

    # open source image
    labels = load_label(src_label_file_name)

    warp_func = None
    try:
        im = imread(src_file_name)
        if len(im.shape) < 2 or len(im.shape) > 3:
            logger.warning(
                "Illegal image file format {0}.".format(src_file_name))
            raise
        elif len(im.shape) == 3:
            # RGB image
            if im.shape[2] != 3:
                logger.warning(
                    "The image must be RGB or monochrome.")
                # csv_data.remove(data)
                raise

        # resize
        h = im.shape[0]
        w = im.shape[1]
        input_size = (w, h)
        if w != width or h != height:
            # resize image
            if mode == 'trimming':
                # trimming mode
                if float(h) / w > float(height) / width:
                    target_h = int(float(w) / width * height)
                    im = im[(h - target_h) // 2:h - (h - target_h) // 2, ::]
                    size_after_mode = (w, target_h)
                else:
                    target_w = int(float(h) / height * width)
                    im = im[::, (w - target_w) // 2:w - (w - target_w) // 2]
                    size_after_mode = (target_w, h)

                warp_func = trim_warp
            elif mode == 'padding':
                # padding mode
                if float(h) / w < float(height) / width:
                    target_h = int(float(height) / width * w)
                    pad = (((target_h - h) // 2, target_h -
                            (target_h - h) // 2 - h), (0, 0))
                    size_after_mode = (w, target_h)
                else:
                    target_w = int(float(width) / height * h)
                    pad = ((0, 0), ((target_w - w) // 2,
                                    target_w - (target_w - w) // 2 - w))
                    size_after_mode = (target_w, h)
                if len(im.shape) == 3:
                    pad = pad + ((0, 0),)
                im = np.pad(im, pad, 'constant')

                warp_func = pad_warp
            im = imresize(im, size=(width, height))

        # change color ch
        if len(im.shape) == 2 and ch == 3:
            # Monochrome to RGB
            im = np.array([im, im, im]).transpose((1, 2, 0))
        elif len(im.shape) == 3 and ch == 1:
            # RGB to monochrome
            im = np.dot(im[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

        # output image
        imsave(image_file_name, im)

    except BaseException:
        logger.warning(
            "Failed to convert %s." % (src_file_name))
        raise

    # create label and region file
    if warp_func is not None:
        labels = [warp_func(label, input_size, size_after_mode)
                  for label in labels]
    grid_w = width // grid_size
    grid_h = height // grid_size

    region_array = np.full(
        (1, grid_h, grid_w, 4), 0.0, dtype=np.float)

    hm = np.zeros((num_class, grid_h, grid_w), dtype=np.float32)

    for label in labels:
        label_rect = ObjectRect(XYWH=label[1:]).clip()

        if label_rect.width() > 0.0 and label_rect.height() > 0.0:
            gx, gy = int(label_rect.centerx() *
                         grid_w), int(label_rect.centery() * grid_h)

            radius = gaussian_radius(
                (math.ceil(label_rect.height() * grid_h), math.ceil(label_rect.width() * grid_w)))

            radius = max(0, int(radius))
            ct = np.array([gx, gy], dtype=np.float32)
            ct_int = ct.astype(np.int32)
            hm[int(label[0])] = draw_umich_gaussian(
                hm[int(label[0])], ct_int, radius)

            region_array[0][gy][gx] = [label_rect.centerx() * grid_w - gx, label_rect.centery() * grid_h - gy,
                                       np.log(label_rect.width() * grid_w), np.log(label_rect.height() * grid_h)]

    np.savetxt(label_file_name, hm.reshape(
        (hm.shape[0] * hm.shape[1], -1)), fmt='%f', delimiter=',')
    np.savetxt(region_file_name, region_array.reshape(
        (region_array.shape[0] * region_array.shape[1], -1)), fmt='%f', delimiter=',')


def create_object_detection_dataset_command(dataset_dict):
    # settings
    source_dir = dataset_dict["inputdir"]
    dest_dir = dataset_dict["outdir"]
    width = int(dataset_dict["width"])
    height = int(dataset_dict["height"])
    mode = dataset_dict["mode"]
    ch = int(dataset_dict["channel"])
    num_class = int(dataset_dict["num_class"])
    grid_size = int(dataset_dict["grid_size"])
    shuffle = dataset_dict["shuffle"]

    if width % grid_size != 0:
        logger.log(99, 'width" must be divisible by grid_size.')
        return
    if height % grid_size != 0:
        logger.log(99, 'height must be divisible by grid_size.')
        return

    dest_csv_file_name = [os.path.join(
        dataset_dict["outdir"], dataset_dict["file1"])]
    if dataset_dict["file2"]:
        dest_csv_file_name.append(os.path.join(
            dataset_dict["outdir"], dataset_dict["file2"]))
    test_data_ratio = int(
        dataset_dict["ratio2"]) if dataset_dict["ratio2"] else 0

    if dataset_dict["inputdir"] == dataset_dict["outdir"]:
        logger.critical("Input directory and output directory are same.")
        return False

    # create file list
    logger.log(99, "Creating file list...")

    file_list = create_file_list(source_dir=source_dir)

    if len(file_list) == 0:
        logger.critical(
            "No image file found in the subdirectory of the input directory.")
        return False

    # create output data
    logger.log(99, "Creating output images...")
    process_dicts = [{'data': data, 'source_dir': source_dir, 'dest_dir': dest_dir, 'width': width,
                      'height': height, 'mode': mode, 'ch': ch, 'num_class': num_class, 'grid_size': grid_size} for data in file_list]
    p = mp.Pool(mp.cpu_count())
    pbar = tqdm.tqdm(total=len(process_dicts))
    for _ in p.imap_unordered(convert_image, process_dicts):
        pbar.update()
    pbar.close()

    file_list = [os.path.join('.', 'data', file) for file in file_list]
    file_list = [file for file in file_list if os.path.exists(
        os.path.join(dest_dir, os.path.splitext(file)[0] + '.png'))]
    if len(file_list) == 0:
        logger.critical("No image and label file created correctly.")
        return False

    logger.log(99, "Creating CSV files...")
    if shuffle:
        random.shuffle(file_list)

    temp_file_list = copy.copy(file_list)
    for png_path in tqdm.tqdm(temp_file_list):
        label_data = np.loadtxt(os.path.join(
            dest_dir, png_path.replace('.png', '_label.csv')), delimiter=',')
        if ~np.any(label_data):
            file_list.remove(png_path)

    csv_data_num = [(len(file_list) * (100 - test_data_ratio)) // 100]
    csv_data_num.append(len(file_list) - csv_data_num[0])
    data_head = 0
    for csv_file_name, data_num in tqdm.tqdm(zip(dest_csv_file_name, csv_data_num), total=len(dest_csv_file_name)):
        if data_num:
            file_list_2 = file_list[data_head:data_head + data_num]
            data_head += data_num

            with open(csv_file_name, 'w') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow(['x:image', 'y:label', 'r:region'])
                for file in file_list_2:
                    base_file_name = os.path.splitext(file)[0]
                    writer.writerow(
                        [file, base_file_name + '_label.csv', base_file_name + '_region.csv'])

    logger.log(99, "Dataset was successfully created.")
    return True
