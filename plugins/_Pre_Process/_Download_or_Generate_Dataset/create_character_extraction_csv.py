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
import csv
import argparse
import zipfile

from nnabla.logger import logger
from nnabla.utils.data_source_loader import download


def func(args):
    path = args.output_dir
    if not os.path.exists(path):
        os.makedirs(path)

    # Create original training set
    logger.log(99, 'Downloading Character Extraction dataset...')

    with zipfile.ZipFile(download('https://zenodo.org/records/16564060/files/character_extraction_dataset.zip')) as zip:
        zip.extractall(path)

    for csv_file in ['train40000\\train40000.csv', 'validation\\validation.csv', 'validation\\validation_small.csv']:
        with open(os.path.join(path, csv_file)) as f:
            reader = csv.reader(f)
            csv_data = [row for row in reader]

        csv_dir, file_name = os.path.split(csv_file)

        for line in csv_data[1:]:
            for col in range(2):
                line[col] = os.path.join(csv_dir, line[col])

        with open(os.path.join(path, file_name), 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerows(csv_data)

    logger.log(99, 'Dataset creation completed successfully.')


def main():
    parser = argparse.ArgumentParser(
        description='CharacterExtraction\n\n' +
        'Download Character Extraction dataset from https://zenodo.org/records/16564060.\n\n',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-o',
        '--output-dir',
        help='path to write NNC dataset CSV format (dir) default=synthetic_data\\character_extraction',
        required=True)
    parser.set_defaults(func=func)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
