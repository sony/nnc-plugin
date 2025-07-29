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
import argparse
import zipfile

from nnabla.logger import logger
from nnabla.utils.data_source_loader import download


def func(args):
    path = args.output_dir

    # Create original training set
    logger.log(99, 'Downloading Flower Food dataset...')

    r = download(
        'https://zenodo.org/records/16563563/files/flower_food_dataset.zip')
    with zipfile.ZipFile(r) as zip:
        zip.extractall(args.output_dir)

    logger.log(99, 'Dataset creation completed successfully.')


def main():
    parser = argparse.ArgumentParser(
        description='FlowerFood\n\n' +
        'Download "Flower Food Dataset" from https://zenodo.org/records/16563563.\n\n',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-o',
        '--output-dir',
        help='path to write NNC dataset CSV format (dir) default=Image\\flower_food',
        required=True)
    parser.set_defaults(func=func)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
