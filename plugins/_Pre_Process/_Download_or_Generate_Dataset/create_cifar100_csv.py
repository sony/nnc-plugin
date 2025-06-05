# Copyright 2022,2023,2024,2025 Sony Group Corporation.
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
import argparse
import io
import os
import pandas as pd

from huggingface_hub import hf_hub_download
from PIL import Image
from tqdm import tqdm
from nnabla.logger import logger


def df_to_csv(base_path, csv_file_name, data_path, df):
    labels = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver',
              'bed', 'bee', 'beetle', 'bicycle', 'bottle',
              'bowl', 'boy', 'bridge', 'bus', 'butterfly',
              'camel', 'can', 'castle', 'caterpillar', 'cattle',
              'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach',
              'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
              'dolphin', 'elephant', 'flatfish', 'forest', 'fox',
              'girl', 'hamster', 'house', 'kangaroo', 'computer_keyboard',
              'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard',
              'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
              'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid',
              'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
              'plain', 'plate', 'poppy', 'porcupine', 'possum',
              'rabbit', 'raccoon', 'ray', 'road', 'rocket',
              'rose', 'sea', 'seal', 'shark', 'shrew',
              'skunk', 'skyscraper', 'snail', 'snake', 'spider',
              'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
              'tank', 'telephone', 'television', 'tiger', 'tractor',
              'train', 'trout', 'tulip', 'turtle', 'wardrobe',
              'whale', 'willow_tree', 'wolf', 'woman', 'worm']

    for label in labels:
        os.makedirs(
            os.path.abspath(os.path.join(base_path, data_path, label)),
            exist_ok=True)

    datalist = []
    for idx, row in tqdm(df.iterrows(), total=len(df), unit='images'):
        relative_path = '/'.join(
            (data_path, labels[row.fine_label], f'{idx}.png'))
        image_path = os.path.abspath(os.path.join(base_path, relative_path))
        Image.open(io.BytesIO(row.img['bytes'])).save(image_path)
        datalist.append([relative_path, row.fine_label])

    columns = ['x:image', 'y:label;' + ';'.join(labels)]
    csv_path = os.path.join(base_path, csv_file_name)
    csv_df = pd.DataFrame(datalist, columns=columns)
    csv_df.to_csv(csv_path, index=False, lineterminator='\n')
    return csv_df


def func(args):
    path = os.path.abspath(args.output_dir)

    # Create original training/test set
    logger.log(99, 'Downloading CIFAR-100 training dataset...')
    _df = pd.read_parquet(
        hf_hub_download(
            repo_id='uoft-cs/cifar100',
            filename='cifar100/train-00000-of-00001.parquet',
            repo_type='dataset'))
    logger.log(99, 'Creating "cifar100_training.csv"... ')
    df_to_csv(path, 'cifar100_training.csv', './training', _df)

    logger.log(99, 'Downloading CIFAR-100 test dataset...')
    _df = pd.read_parquet(
        hf_hub_download(
            repo_id='uoft-cs/cifar100',
            filename='cifar100/test-00000-of-00001.parquet',
            repo_type='dataset'))
    logger.log(99, 'Creating "cifar100_test.csv"... ')
    df_to_csv(path, 'cifar100_test.csv', './validation', _df)

    logger.log(99, 'Dataset creation completed successfully.')


def main():
    parser = argparse.ArgumentParser(
        description='CIFAR100\n\n' +
        'Download CIFAR-100 dataset from huggingface.\n\n',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-o',
        '--output-dir',
        help='path to write NNC dataset CSV format (dir) default=CIFAR100',
        required=True)
    parser.set_defaults(func=func)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
