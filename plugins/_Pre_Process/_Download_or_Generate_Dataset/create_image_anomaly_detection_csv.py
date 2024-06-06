# Copyright 2024 Sony Group Corporation.
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
import argparse
import math
import csv
import tqdm
import numpy as np
from scipy import signal
from PIL import Image, ImageDraw


from nnabla.logger import logger
from nnabla.utils.image_utils import imsave


def makedirs(dir):
    try:
        os.makedirs(dir)
    except:
        pass


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


def create_csv(target_dir, dataset_name, num_data, with_abnormal_data):
    # CSV header

    csv_data = [['x:image']]
    if with_abnormal_data:
        csv_data[0].append('y:abnormal')

    # Create output dir
    makedirs(os.path.join(target_dir, dataset_name))

    # Create samples
    image_size = 224
    minimum_object_size = 16
    maximum_object_size = 48
    max_aspect_ratio = 3.0

    # Create output dir
    makedirs(os.path.join(target_dir, dataset_name))

    for data_index in tqdm.tqdm(range(num_data)):
        img = np.random.randn(3, (image_size + 1) *
                              image_size) * (0.2 + np.random.rand() * 0.2)
        for i in range(img.shape[0]):
            b, a = signal.butter(
                1, 0.005 + np.random.rand() * 0.01, btype='low')
            img[i] = signal.filtfilt(b, a, img[i])
        img = img.reshape(3, image_size + 1,
                          image_size)[:, 1:, :] + 0.4 + np.random.rand() * 0.2

        if with_abnormal_data:
            abnormal = data_index % 2
            if abnormal:
                abnormal_type = np.random.randint(5)
                cx, cy = np.random.rand() * image_size, np.random.rand() * image_size

                size = np.exp(np.random.rand() * np.log(maximum_object_size /
                                                        minimum_object_size)) * minimum_object_size
                aspect_ratio = np.sqrt(
                    np.exp(np.random.rand() * np.log(max_aspect_ratio)))
                w, h = size * aspect_ratio, size / aspect_ratio
                image = Image.new('L', (image_size, image_size), 0)
                draw = ImageDraw.Draw(image)
                coords = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
                if abnormal_type == 0:
                    # Ellipse
                    draw.ellipse(coords, 255)
                elif abnormal_type < 4:
                    # Triangle, Rectangle, Pentagon
                    draw_polygon(draw, abnormal_type + 2, coords, 255)
                else:
                    # Line
                    if np.random.randint(2) == 1:
                        coords[0], coords[2] = coords[2], coords[0]
                    draw.line(coords, 255, np.random.randint(2, 5))

                img += (np.array(image) / 255.0).reshape(1, image_size, image_size) * \
                    (0.05 + np.random.rand() * 0.05) * \
                    (np.random.randint(2) * 2 - 1)

        img = np.clip(img, 0.0, 1.0)

        # Save image
        data_file_name = os.path.join(
            target_dir, dataset_name, str(data_index) + '.png')
        imsave(data_file_name, img, channel_first=True)
        if with_abnormal_data:
            csv_data.append([data_file_name, str(abnormal)])
        else:
            csv_data.append([data_file_name])

    # Save CSV
    with open(os.path.join(target_dir, dataset_name + '.csv'), 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(csv_data)


def func(args):
    path = args.output_dir
    np.random.seed(0)

    # Create training dataset (normal data only)
    logger.log(99, 'Creating training dataset (unsupervised) ... ')
    create_csv(path, "training_unsupervised", 300, False)

    # Create training dataset (normal and abnormal data)
    logger.log(99, 'Creating training dataset (supervised) ... ')
    create_csv(path, "training_labeled", 300, True)

    # Create validation dataset (normal and abnormal data)
    logger.log(99, 'Creating validation dataset ... ')
    create_csv(path, "validation_labeled", 100, True)

    logger.log(99, 'Dataset creation completed successfully.')


def main():
    parser = argparse.ArgumentParser(
        description='SyhtheticImage_AnomalyDetection\n\n' +
        'Create "Syhthetic image anomaly detection" dataset.\n\n',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-o',
        '--output-dir',
        help='path to write NNC dataset CSV format (dir) default=synthetic_data\\image_anomaly_detection',
        required=True)
    parser.set_defaults(func=func)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
