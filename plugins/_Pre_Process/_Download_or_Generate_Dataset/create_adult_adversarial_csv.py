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
import sys
from tqdm import tqdm
import unicodedata
import subprocess
from nnabla.logger import logger
from nnabla.utils.data_source_loader import download


def func(args):
    path = args.output_dir
    if not os.path.exists(path):
        os.makedirs(path)

    subset = 'data'  # adult data set
    # Download dataset
    logger.log(99, 'Downloading adult.{} dataset...'.format(subset))

    file_name = os.path.join(path, f'adult.{subset}')
    data = download(
        f'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.{subset}').read()
    with open(file_name, 'wb') as f:
        f.write(data)

    # Create original dataset file
    csv_file_name = 'adult_original_{}.csv'.format(subset)
    logger.log(99, 'Creating "{}"... '.format(csv_file_name))

    with open(file_name) as f:
        reader = csv.reader(f)
        data = [[col.strip() for col in row]
                for row in reader if
                len(row) > 0 and any(e in [' White', ' Black'] for e in row) and not ' ?' in row]  # Restrict races to White and Black
    if subset == 'test':
        # remove 1st line
        data = data[1:]

        # remove '.' from label
        for row in data:
            row[14] = row[14].replace('.', '')

    data.insert(0, ['age', 'workclass', 'fnlwgt', 'wducation', 'education-num', 'marital-status', 'occupation',
                    'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                    'native-country', 'label'])

    with open(os.path.join(path, csv_file_name), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)

    # Create text classification dataset
    logger.log(99, 'Creating "adversarial_adult_{}.csv"... '.format(subset))
    command = [sys.executable, os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '_Create_Dataset',
                                            'simple_tabular_data.py'), '-i', os.path.join(path, csv_file_name),
               '-b', 'label', '-z', 'sex,race', '-t', '-{}'.format(
        'p' if subset == 'test' else 'r'), os.path.join(path, 'preprocessing_parameters.csv'), '-o', path, '-g',
        'log.txt', '-f1', 'adv_adult_train.csv'.format(subset), '-r1',
        '50', '-f2', 'adv_adult_test.csv', '-r2', '50']
    p = subprocess.call(command)
    os.remove(file_name)
    os.remove(csv_file_name)

    logger.log(99, 'Dataset creation completed successfully.')


def main():
    parser = argparse.ArgumentParser(
        description='Adult_ADV\n\n' +
                    'Download and Preprocess the UCI Adult Data Set.\n\n',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-o',
        '--output-dir',
        help='path to write NNC dataset CSV format (dir) default=Structured\\Classification\\adult_adv',
        required=True)
    parser.set_defaults(func=func)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
