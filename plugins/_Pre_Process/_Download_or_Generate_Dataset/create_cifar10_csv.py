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

from concurrent import futures
from huggingface_hub import hf_hub_download
from PIL import Image
from tqdm import tqdm
from nnabla.logger import logger


def df_to_csv(base_path, csv_file_name, data_path, df):
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
              'dog', 'frog', 'horse', 'ship', 'truck']

    for label in labels:
        os.makedirs(
            os.path.abspath(os.path.join(base_path, data_path, label)),
            exist_ok=True)

    datalist = [None] * len(df)
    with futures.ThreadPoolExecutor() as executor:
        def _save_image(idx, row):
            rel_path = '/'.join((data_path, labels[row.label], f'{idx}.png'))
            abs_path = os.path.abspath(os.path.join(base_path, rel_path))
            Image.open(io.BytesIO(row.img['bytes'])).save(abs_path)
            return idx, [rel_path, row.label]

        futurelist = [
            executor.submit(_save_image, idx, row)
            for idx, row in tqdm(df.iterrows(), total=len(df), unit='issues')]

        for future in tqdm(futures.as_completed(futurelist),
                           total=len(df), unit='images'):
            idx, result = future.result()
            datalist[idx] = result

    columns = ['x:image', 'y:label;' + ';'.join(labels)]
    csv_path = os.path.join(base_path, csv_file_name)
    csv_df = pd.DataFrame(datalist, columns=columns)
    csv_df.to_csv(csv_path, index=False, lineterminator='\n')
    return csv_df


def func(args):
    path = os.path.abspath(args.output_dir)

    # Create original training/test set
    logger.log(99, 'Downloading CIFAR-10 training dataset...')
    _df = pd.read_parquet(
        hf_hub_download(
            repo_id='uoft-cs/cifar10',
            filename='plain_text/train-00000-of-00001.parquet',
            repo_type='dataset'))
    logger.log(99, 'Creating "cifar10_training.csv"... ')
    df_to_csv(path, 'cifar10_training.csv', './training', _df)

    logger.log(99, 'Downloading CIFAR-10 test dataset...')
    _df = pd.read_parquet(
        hf_hub_download(
            repo_id='uoft-cs/cifar10',
            filename='plain_text/test-00000-of-00001.parquet',
            repo_type='dataset'))
    logger.log(99, 'Creating "cifar10_test.csv"... ')
    df_to_csv(path, 'cifar10_test.csv', './validation', _df)

    logger.log(99, 'Dataset creation completed successfully.')


def main():
    parser = argparse.ArgumentParser(
        description='CIFAR10\n\n' +
        'Download CIFAR-10 dataset from huggingface.\n\n',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-o',
        '--output-dir',
        help='path to write NNC dataset CSV format (dir) default=CIFAR10',
        required=True)
    parser.set_defaults(func=func)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
