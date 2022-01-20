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
from nnabla.logger import logger
from object_detection_utils.create_object_detection_yolov2 import create_image_and_label
from nnabla.utils.cli.create_object_detection_dataset import create_object_detection_dataset_command


def func(args):
    path = args.output_dir

    # Create image and labels
    logger.log(99, 'Creating image and label ... ')
    create_image_and_label(path, "original", 100000)

    class Args:
        pass
    args = Args()
    args.sourcedir = os.path.join(path, 'original')
    args.outdir = path
    args.num_class = '4'
    args.channel = '1'
    args.width = '112'
    args.height = '112'
    args.num_anchor = '5'
    args.grid_size = '16'
    args.mode = 'resize'
    args.shuffle = 'false'
    args.file1 = 'training.csv'
    args.ratio1 = '90'
    args.file2 = 'validation.csv'
    args.ratio2 = '10'
    create_object_detection_dataset_command(args)

    logger.log(99, 'Dataset creation completed successfully.')


def main():
    parser = argparse.ArgumentParser(
        description='SyntheticImage_ObjectDetection\n\n' +
        'Create "Synthetic image object detection" dataset.\n\n',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-o',
        '--output-dir',
        help='path to write NNC dataset CSV format (dir) default=synthetic_data\\object_detection',
        required=True)
    parser.set_defaults(func=func)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
