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
import argparse
import shutil
import zipfile
import glob
import random
import csv
from PIL import Image
from tqdm import tqdm

from nnabla.logger import logger
from nnabla.utils.data_source_loader import download


def create_sr_sample_dataset(path, csv_file_name, files, offset=0):
    logger.log(99, f'Creating {csv_file_name}...')
    table = [['x:low_res_image', 'y:hi_res_image']]
    for i, file in enumerate(tqdm(files)):
        img = Image.open(file)
        hi_res_file_name = os.path.join(
            path, 'images', 'hi_res', f'{i + offset}.png')
        img.save(os.path.join(path, hi_res_file_name))
        img = img.resize((112, 112), Image.LANCZOS)
        lo_res_file_name = os.path.join(
            path, 'images', 'lo_res', f'{i + offset}.png')
        img.save(os.path.join(path, lo_res_file_name))
        table.append([lo_res_file_name, hi_res_file_name])

    csv_file_name = os.path.join(path, csv_file_name)
    with open(csv_file_name, 'w', newline="\n") as f:
        writer = csv.writer(f)
        writer.writerows(table)


def func(args):
    path = args.output_dir
    original_dataset_path = os.path.join(path, 'flower_food_dataset')

    # Create original training set
    logger.log(99, 'Downloading Flower Food dataset...')

    r = download(
        'https://zenodo.org/records/16563563/files/flower_food_dataset.zip')
    with zipfile.ZipFile(r) as zip:
        zip.extractall(original_dataset_path)

    image_files = glob.glob(os.path.join(original_dataset_path, 'flower', '*.png')) + \
        glob.glob(os.path.join(original_dataset_path, 'food', '*.png'))
    random.shuffle(image_files)

    try:
        os.makedirs(os.path.join(path, 'images'))
        os.makedirs(os.path.join(path, 'images', 'lo_res'))
        os.makedirs(os.path.join(path, 'images', 'hi_res'))
    except:
        pass
    create_sr_sample_dataset(path, os.path.join(
        path, 'flower_food_sr_train.csv'), image_files[:380])
    create_sr_sample_dataset(path, os.path.join(
        path, 'flower_food_sr_test.csv'), image_files[380:], 380)
    shutil.rmtree(original_dataset_path)

    logger.log(99, 'Dataset creation completed successfully.')


def main():
    parser = argparse.ArgumentParser(
        description='FlowerFoodSR\n\n' +
        'Download "Flower Food Dataset" from https://zenodo.org/records/16563563.\n\n',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-o',
        '--output-dir',
        help='path to write NNC dataset CSV format (dir) default=Image\\flower_food_sr',
        required=True)
    parser.set_defaults(func=func)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
