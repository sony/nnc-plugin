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
import subprocess
from nnabla.logger import logger
from object_detection_utils.create_object_detection_yolov2 import create_image_and_label, makedirs


def func(args):
    path = args.output_dir

    # Create image and labels
    logger.log(99, 'Creating image and label ... ')
    create_image_and_label(path, "original", 100000)

    # Create Csv dataset
    makedirs(path)
    command = ['Python', os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '_Create_Dataset', 'object_detection_centernet.py'), '-i', os.path.join(path, 'original'), '-o', path,
               '-nc', '4', '-ch', '1', '-w', '112', '-ht', '112', '-g', '4', '-m', 'resize', '-f1', 'training_for_centernet.csv', '-r1', '90', '-f2', 'validation_for_centernet.csv', '-r2', '10']
    p = subprocess.call(command)

    # logger.log(99, 'Dataset creation completed successfully.')


def main():
    parser = argparse.ArgumentParser(
        description='SyntheticImage_ObjectDetection_CenterNet\n\n' +
        'Create "Synthetic image object detection (CenterNet)" dataset.\n\n',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-o',
        '--output-dir',
        help='path to write NNC dataset CSV format (dir) default=synthetic_data\\object_detection_for_centernet',
        required=True)
    parser.set_defaults(func=func)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
