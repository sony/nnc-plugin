# Copyright 2021 Sony Group Corporation.
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

from nnabla import logger
from nnabla.utils.cli.create_object_detection_dataset import create_object_detection_dataset_command


def func(args):
    args.sourcedir = args.input_dir
    args.outdir = args.output_dir
    args.num_anchor = args.anchor
    args.file1 = args.output_file1
    args.file2 = args.output_file2
    args.shuffle = 'true' if args.shuffle else 'false'
    create_object_detection_dataset_command(args)
    logger.log(99, 'Dataset creation completed successfully.')


def main():
    parser = argparse.ArgumentParser(
        description='Object Detection (Yolo v2 format)\n\n' +
        'Convert Yolo v2 object detection dataset format to NNC dataset CSV format.\n\n',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-i',
        '--input-dir',
        help='path where Yolo v2 format data is placed (dir)',
        required=True)
    parser.add_argument(
        '-o',
        '--output-dir',
        help='path to write NNC dataset CSV format (dir)',
        required=True)
    parser.add_argument(
        '-n',
        '--num-class',
        help='number of class (int) default=10',
        required=True)
    parser.add_argument(
        '-c',
        '--channel',
        help='number of output color channels (int) default=3',
        required=True)
    parser.add_argument(
        '-w', '--width', help='width (int) default=256', required=True)
    parser.add_argument(
        '-g', '--height', help='height (int) default=256', required=True)
    parser.add_argument(
        '-a',
        '--anchor',
        help='number of anchor (int) default=5',
        required=True)
    parser.add_argument(
        '-d',
        '--grid-size',
        help='size of the grid in pixels (int) default=16',
        required=True)
    parser.add_argument(
        '-m',
        '--mode',
        help='shaping mode (option:trimming,padding,resize) default=trimming',
        required=True)
    parser.add_argument(
        '-s',
        '--shuffle',
        help='shuffle mode (option:true,false) default=true',
        required=True)
    parser.add_argument(
        '-f1',
        '--output_file1',
        help='output file name 1 (csv) default=train.csv',
        required=True,
        default='train.csv')
    parser.add_argument(
        '-r1',
        '--ratio1',
        help='output file ratio 1 (int) default=100',
        required=True)
    parser.add_argument(
        '-f2',
        '--output_file2',
        help='output file name 2 (csv) default=test.csv',
        default='test.csv')
    parser.add_argument(
        '-r2',
        '--ratio2',
        help='output file ratio 2 (int) default=0',
        default=0)
    parser.set_defaults(func=func)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
