# Copyright 2022 Sony Group Corporation.
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
import os
import math
import csv
import tqdm
import numpy as np
from PIL import Image, ImageDraw
import subprocess


def makedirs(dir):
    try:
        os.makedirs(dir)
    except:
        pass


def create_image_and_label(target_dir, dataset_name, num_data):
    image_size = 112
    grid_size = 7
    minimum_object_size = 16
    maximum_object_size = 112
    max_aspect_ratio = 2.0
    max_num_object = 10
    num_object_type = 4  # Ellipse, Triangle, Rectangle, Pentagon

    # Create output dir
    makedirs(os.path.join(target_dir, dataset_name))

    def create_object(bgcolor):
        cx, cy = np.random.rand() * image_size, np.random.rand() * image_size

        size = np.exp(np.random.rand() * np.log(maximum_object_size /
                      minimum_object_size)) * minimum_object_size
        aspect_ratio = np.sqrt(
            np.exp(np.random.rand() * np.log(max_aspect_ratio)))
        w, h = size * aspect_ratio, size / aspect_ratio

        object_type = np.random.randint(num_object_type)
        while True:
            color = np.random.randint(256)
            if np.abs(color - bgcolor) >= 32:
                break

        # print(object_type, color, cx, cy, w, h)
        return {'type': object_type, 'color': color, 'cx': cx, 'cy': cy, 'w': w, 'h': h}

    def delete_overlap(objects):
        result = []
        for i, object in enumerate(objects):
            area = object['w'] * object['h']
            area_bak = area
            left, right = object['cx'] - object['w'] / \
                2, object['cx'] + object['w'] / 2
            top, bottom = object['cy'] - object['h'] / \
                2, object['cy'] + object['h'] / 2
            int_x, int_y = int(
                object['cx'] / grid_size), int(object['cy'] / grid_size)
            for object2 in objects[i + 1:]:

                int_x2, int_y2 = int(
                    object2['cx'] / grid_size), int(object2['cy'] / grid_size)
                if int_x == int_x2 and int_y == int_y2:
                    area = 0
                    break
                left2, right2 = object2['cx'] - object2['w'] / \
                    2, object2['cx'] + object2['w'] / 2
                top2, bottom2 = object2['cy'] - object2['h'] / \
                    2, object2['cy'] + object2['h'] / 2
                w = np.min([right, right2]) - np.max([left, left2])
                h = np.min([bottom, bottom2]) - np.max([top, top2])
                if w > 0 and h > 0:
                    area -= w * h
            if area / area_bak >= 0.2:
                result.append(object)
        return result

    def draw_polygon(draw, num_vertices, coords, color):
        rad = math.pi * 2.0 * np.random.rand()
        x, y = [], []
        for vi in range(num_vertices):
            x.append(math.cos(rad))
            y.append(math.sin(rad))
            rad += math.pi * 2.0 / num_vertices

        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        y = (y - np.min(y)) / (np.max(y) - np.min(y))
        draw.polygon([(x_ * (coords[2] - coords[0]) + coords[0], y_ *
                     (coords[3] - coords[1]) + coords[1]) for x_, y_ in zip(x, y)], color)

    # Create samples
    np.random.seed(0)
    for data_index in tqdm.tqdm(range(num_data)):
        sub_dir = os.path.join('.', dataset_name, str(
            int(data_index / 1000)).zfill(4))
        if not os.path.exists(os.path.join(target_dir, sub_dir)):
            makedirs(os.path.join(target_dir, sub_dir))

        # Create object
        bgcolor = np.random.randint(256)
        objects = [create_object(bgcolor) for _ in range(
            int(np.exp(np.random.rand() * np.log(max_num_object))))]
        objects = delete_overlap(objects)

        # Create image
        data_file_name = os.path.join(sub_dir, str(data_index) + '.png')
        image = Image.new('L', (image_size, image_size), bgcolor)
        draw = ImageDraw.Draw(image)
        for object in objects:
            coords = [object['cx'] - object['w']/2, object['cy'] - object['h'] /
                      2, object['cx'] + object['w']/2, object['cy'] + object['h']/2]
            if object['type'] == 0:
                # Ellipse
                draw.ellipse(coords, object['color'])
            elif object['type'] < 4:
                draw_polygon(draw, object['type'] + 2, coords, object['color'])
            else:
                # Rectangle
                draw.rectangle(coords, object['color'])
                raise

        image.save(os.path.join(target_dir, data_file_name))

        # Create label txt
        txt_file_name = os.path.join(sub_dir, str(data_index) + '.txt')
        with open(os.path.join(target_dir, txt_file_name), mode='w') as f:
            for object in objects:
                f.write('{} {} {} {} {}\n'.format(object['type'], object['cx'] / image_size,
                        object['cy'] / image_size, object['w'] / image_size, object['h'] / image_size))
