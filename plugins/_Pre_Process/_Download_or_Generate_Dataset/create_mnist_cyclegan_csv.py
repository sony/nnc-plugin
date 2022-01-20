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

from nnabla.logger import logger
from create_mnist_csv import func as create_mnist_dataset


def func(args):
    path = args.output_dir

    csv_file_names = [os.path.join(path, file_name) for file_name in [
        'mnist_training.csv', 'small_mnist_4or9_test.csv']]

    if not os.path.exists(csv_file_names[0]):
        # create mnist dataset first
        create_mnist_dataset(args)

    for i, csv_file_name in enumerate(csv_file_names):
        logger.log(99, f'Loading {csv_file_name}.')
        with open(csv_file_name) as f:
            reader = csv.reader(f)
            table = [row for row in reader]
            header = table.pop(0)

        # Extract table for each 4 and 9
        for i2 in range(2):
            if i == 0:
                out_file_name = f'mnist_for_cyclegan_training_{4 if i2 == 0 else 9}.csv'
                out_table = [[row[0]]
                             for row in table if row[1] == str(4 if i2 == 0 else 9)]
                out_header = ['x2' if i2 == 1 else 'x']
            else:
                out_table = [[row[0]] for row in table if row[1] == str(i2)]
                if i2 == 1:
                    out_file_name = 'mnist_for_cyclegan_test.csv'
                    out_header = ['x', 'x2']
                    out_table = [x + x2 for x,
                                 x2 in zip(out_table_, out_table)]
                else:
                    out_file_name = ''
                    out_table_ = out_table

            if out_file_name:
                logger.log(99, f'Creating {out_file_name}.')
                with open(os.path.join(path, out_file_name), 'w') as f:
                    writer = csv.writer(f, lineterminator='\n')
                    writer.writerow(out_header)
                    writer.writerows(out_table)

    logger.log(99, 'Dataset creation completed successfully.')


def main():
    parser = argparse.ArgumentParser(
        description='MNISTCycleGAN\n\n' +
        'Download MNIST dataset from dl.sony.com (original file is from http://yann.lecun.com/exdb/mnist/).\n\n',
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
