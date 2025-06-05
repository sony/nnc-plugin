# Copyright 2021,2022,2023,2024,2025 Sony Group Corporation.
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
import math
import numpy as np
import os
import pandas as pd
import random

from huggingface_hub import hf_hub_download
from PIL import Image
from tqdm import tqdm
from nnabla.logger import logger


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


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


def df_to_csv(base_path, csv_file_name, data_path, df):

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

    csv_df = pd.DataFrame(datalist, columns=['x', 'y'])
    return put_csv(os.path.join(base_path, csv_file_name), csv_df)


def put_csv(path, df):
    columns = ['x:image', 'y:label;' + ';'.join(labels)]
    if df.shape[1] == 3:
        columns.append('original_label')

    df.to_csv(path, index=False, header=columns, lineterminator='\n')
    return df


def shuffle_label(label_df, shuffle_rate):
    num_class = label_df.max() + 1

    shuffle_df = pd.Series(dtype=int)
    for label in range(num_class):
        num_wrong_per_class = round((label_df == label).sum() * shuffle_rate)
        num_correct_per_class = (label_df == label).sum() - num_wrong_per_class

        # generate `num_wrong_per_class` wrong labels
        num_repeat = math.ceil(num_wrong_per_class / (num_class - 1))
        artificial_label = np.tile(
            [i for i in range(num_class) if i != label],
            num_repeat)[:num_wrong_per_class]
        np.random.shuffle(artificial_label)

        # concatenate with correct labels
        update_labels = np.insert(
            artificial_label, 0, np.full(num_correct_per_class, label))
        np.random.shuffle(update_labels)

        # add the updated labels to shuffle_df
        shuffle_df = pd.concat([
            shuffle_df,
            pd.Series(update_labels, index=label_df[label_df == label].index)])
    return shuffle_df.sort_index()


def main(args):
    path = os.path.abspath(os.path.dirname(__file__))

    # Create original training set
    logger.log(99, "Downloading CIFAR-100 training dataset...")
    _df = pd.read_parquet(
        hf_hub_download(
            repo_id='uoft-cs/cifar100',
            filename='cifar100/train-00000-of-00001.parquet',
            repo_type='dataset'))
    logger.log(99, 'Creating "cifar100_training.csv"... ')
    train_csv_df = df_to_csv(path, 'cifar100_training.csv', './training', _df)

    if args.label_shuffle:
        logger.log(99, 'Creating "cifar100_training_shuffle.csv"... ')
        label_df = shuffle_label(train_csv_df['y'].copy(), args.shuffle_rate)

        num = (label_df != train_csv_df['y']).sum()
        logger.log(99, f'{num} labels are shuffled.')

        train_csv_df.insert(1, 'shuffle', label_df)
        put_csv(os.path.join(path, 'cifar100_training_shuffle.csv'),
                train_csv_df)

    logger.log(99, 'Downloading CIFAR-100 test dataset...')
    _df = pd.read_parquet(
        hf_hub_download(
            repo_id='uoft-cs/cifar100',
            filename='cifar100/test-00000-of-00001.parquet',
            repo_type='dataset'))
    logger.log(99, 'Creating "cifar100_test.csv"... ')
    df_to_csv(path, 'cifar100_test.csv', './validation', _df)

    logger.log(99, "Dataset creation completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label_shuffle",
        action="store_true",
        help="generate label shuffled dataset")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle_rate", type=float, default=0.1)
    args = parser.parse_args()

    set_seed(args.seed)
    main(args)
