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
import io
import os
import argparse
import numpy as np
import pandas as pd

from concurrent import futures
from huggingface_hub import hf_hub_download
from PIL import Image
from tqdm import tqdm
from nnabla.logger import logger


def df_to_csv(base_path, csv_file_name, data_path, df):
    for label in df.label.unique():
        os.makedirs(
            os.path.abspath(os.path.join(base_path, data_path, str(label))),
            exist_ok=True)

    datalist = [None] * len(df)
    with futures.ThreadPoolExecutor() as executor:
        def _save_image(idx, row):
            rel_path = '/'.join((data_path, str(row.label), f'{idx}.png'))
            abs_path = os.path.abspath(os.path.join(base_path, rel_path))
            Image.open(io.BytesIO(row.image['bytes'])).save(abs_path)
            return idx, [rel_path, row.label]

        futurelist = [
            executor.submit(_save_image, idx, row)
            for idx, row in tqdm(df.iterrows(), total=len(df), unit='issues')]

        for future in tqdm(futures.as_completed(futurelist),
                           total=len(df), unit='images'):
            idx, result = future.result()
            datalist[idx] = result

    csv_path = os.path.join(base_path, csv_file_name)
    csv_df = pd.DataFrame(datalist, columns=['x:image', 'y:label'])
    csv_df.to_csv(csv_path, index=False, lineterminator='\n')
    return csv_df


def build_onehot_df(df):
    columns = ['x:image']
    columns.extend([f'y__{i}:{i}' for i in range(10)])

    csv_df = pd.get_dummies(df, dtype=int, columns=['y:label'])
    csv_df = csv_df.set_axis(columns, axis=1, copy=False)
    return csv_df


def func(args):
    path = os.path.abspath(args.output_dir)

    # Create original training/test set
    logger.log(99, 'Downloading MNIST training set images...')
    _df = pd.read_parquet(
        hf_hub_download(
            repo_id='ylecun/mnist',
            filename='mnist/train-00000-of-00001.parquet',
            repo_type='dataset'))
    logger.log(99, 'Creating "mnist_training.csv"... ')
    train_csv_df = df_to_csv(path, 'mnist_training.csv', './training', _df)

    logger.log(99, 'Downloading MNIST test set images...')
    _df = pd.read_parquet(
        hf_hub_download(
            repo_id='ylecun/mnist',
            filename='mnist/test-00000-of-00001.parquet',
            repo_type='dataset'))
    logger.log(99, 'Creating "mnist_test.csv"... ')
    test_csv_df = df_to_csv(path, 'mnist_test.csv', './validation', _df)

    # Create one-hot training/test set
    logger.log(99, 'Creating "mnist_training_onehot.csv"... ')
    onehot_train_csv_df = build_onehot_df(train_csv_df)
    onehot_train_csv_df.to_csv(
        os.path.join(path, 'mnist_training_onehot.csv'),
        index=False,
        lineterminator='\n')

    logger.log(99, 'Creating "mnist_test_onehot.csv"... ')
    onehot_test_csv_df = build_onehot_df(test_csv_df)
    onehot_test_csv_df.to_csv(
        os.path.join(path, 'mnist_test_onehot.csv'),
        index=False,
        lineterminator='\n')

    # Create 100 data training set for semi-supervised learning
    logger.log(99, 'Creating "mnist_training_100.csv"... ')
    tens_train_df = pd.DataFrame()
    for label in train_csv_df['y:label'].unique():
        first_ten_df = (train_csv_df[train_csv_df['y:label'] == label])[:10]
        tens_train_df = pd.concat([tens_train_df, first_ten_df])
    tens_train_df.sort_index().to_csv(
        os.path.join(path, 'mnist_training_100.csv'),
        index=False,
        lineterminator='\n')

    # Create unlabeled training set for semi-supervised learning
    logger.log(99, 'Creating "mnist_training_unlabeled.csv"... ')
    unlabel_train_df = train_csv_df.drop(columns='y:label')
    unlabel_train_df = unlabel_train_df.set_axis(
        ['xu:image'], axis=1, copy=False)
    unlabel_train_df.to_csv(
        os.path.join(path, 'mnist_training_unlabeled.csv'),
        index=False,
        lineterminator='\n')

    # Create small training/test set
    logger.log(99, 'Creating "small_mnist_4or9_training.csv"... ')
    small_train_df = pd.concat([
        train_csv_df[train_csv_df['y:label'] == 4][:750],
        train_csv_df[train_csv_df['y:label'] == 9][:750]
    ])
    small_train_df['y:label'] = (small_train_df['y:label'] == 9).astype(int)
    small_train_df.sort_index(inplace=True)
    small_train_df = small_train_df.set_axis(
        ['x:image', 'y:label;4;9'], axis=1, copy=False)
    small_train_df.to_csv(
        os.path.join(path, 'small_mnist_4or9_training.csv'),
        index=False,
        lineterminator='\n')

    logger.log(99, 'Creating "small_mnist_4or9_test.csv"... ')
    small_test_df = pd.concat([
        test_csv_df[test_csv_df['y:label'] == 4][:250],
        test_csv_df[test_csv_df['y:label'] == 9][:250]
    ])
    small_test_df['y:label'] = (small_test_df['y:label'] == 9).astype(int)
    small_test_df.sort_index(inplace=True)
    small_test_df = small_test_df.set_axis(
        ['x:image', 'y:label;4;9'], axis=1, copy=False)
    small_test_df.to_csv(
        os.path.join(path, 'small_mnist_4or9_test.csv'),
        index=False,
        lineterminator='\n')

    # Create small test set with initial memory
    logger.log(99, 'Creating "small_mnist_4or9_test_w_initmemory.csv"... ')
    memory_size = 256
    zeros = pd.DataFrame(
        np.zeros((len(small_test_df), memory_size)),
        columns=[f'c__{i}' for i in range(memory_size)],
        index=small_test_df.index)
    mem_test_df = pd.concat([small_test_df, zeros], axis=1)
    mem_test_df.to_csv(
        os.path.join(path, 'small_mnist_4or9_test_w_initmemory.csv'),
        index=False,
        lineterminator='\n')
    logger.log(99, 'Dataset creation completed successfully.')


def main():
    parser = argparse.ArgumentParser(
        description='MNIST\n\n' +
        'Download MNIST dataset from huggingface.\n\n',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-o',
        '--output-dir',
        help='path to write NNC dataset CSV format (dir) default=mnist',
        required=True)
    parser.set_defaults(func=func)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
