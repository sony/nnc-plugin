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
import csv
from tqdm import tqdm
import unicodedata
import subprocess

from sklearn.datasets import fetch_20newsgroups

from nnabla.logger import logger


def func(args):
    path = args.output_dir
    if not os.path.exists(path):
        os.makedirs(path)

    for subset in ['train', 'test']:
        # Download dataset
        logger.log(99, 'Downloading 20newsgroups {} dataset...'.format(subset))

        dataset = fetch_20newsgroups(data_home=path, subset=subset)

        # Create original dataset file
        logger.log(
            99, 'Creating "20newsgroups_original_{}.csv"... '.format(subset))
        csv_data = []
        for text, target in zip(tqdm(dataset.data), dataset.target):
            csv_data.append(["".join(c if unicodedata.category(
                c)[0] != "C" and c != '"' else " " for c in text[:2048]), target])
        with open(os.path.join(path, "20newsgroups_original_{}.csv").format(subset), 'w', encoding="utf-8") as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(["x:text", "y:target;{}".format(
                ';'.join(dataset.target_names))])
            writer.writerows(csv_data)

        # Create text classification dataset
        logger.log(99, 'Creating "20newsgroups_{}.csv"... '.format(subset))
        command = ['Python', os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '_Create_Dataset', 'simple_text_classification.py'), '-i', os.path.join(path, '20newsgroups_original_{}.csv'.format(subset)), '-E', 'utf-8',
                   '-l', '256', '-w', '16384', '-m', '8', '-n', '-{}'.format('d' if subset == 'test' else 'e'), os.path.join(path, 'index.csv'), '-o', path, '-g', 'log.txt', '-f1', '20newsgroups_{}.csv'.format(subset), '-r1', '100']
        p = subprocess.call(command)

    logger.log(99, 'Dataset creation completed successfully.')


def main():
    parser = argparse.ArgumentParser(
        description='20newsgroups\n\n' +
        'Download 20 Newsgroups dataset from http://qwone.com/~jason/20Newsgroups/.\n\n',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-o',
        '--output-dir',
        help='path to write NNC dataset CSV format (dir) default=NLP\\Classification\\20newsgroups',
        required=True)
    parser.set_defaults(func=func)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
