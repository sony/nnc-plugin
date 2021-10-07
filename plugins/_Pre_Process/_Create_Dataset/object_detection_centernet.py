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
import argparse
import os

from nnabla.logger import logger

import object_detection_centernet_util


def main():
    parser = argparse.ArgumentParser(
        description='Object Detection (for CenterNet from Yolo v2 format)\n\n'
        'Convert Yolo v2 object detection dataset format to NNC dataset CSV format for CenterNet.\n\n',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '-i',
        '--input_dir',
        help='source directory with souce image and label files(dir)',
        default=None,
        required=True
    )
    parser.add_argument(
        '-o',
        '--output_dir',
        help='output directory(dir)',
        default=None,
        required=True
    )
    parser.add_argument(
        '-nc',
        '--num_class',
        help='number of object classes (int) default=4',
        default=4,
        type=int,
        required=True
    )
    parser.add_argument(
        '-ch',
        '--channel',
        help='number of output color channels (int) default=1',
        default=1,
        type=int,
        required=True
    )
    parser.add_argument(
        '-w',
        '--width',
        help='width of output image (int) default=112',
        default=112,
        type=int,
        required=True
    )
    parser.add_argument(
        '-ht',
        '--height',
        help='height of output image (int) default=112',
        default=112,
        type=int,
        required=True
    )
    parser.add_argument(
        '-g',
        '--grid_size',
        help='width and height of detection grid (int) default=4',
        default=4,
        type=int,
        required=True
    )
    parser.add_argument(
        '-m',
        '--mode',
        help='shaping mode (option:resize,trimming,padding) default=trimming',
        default='resize',
        required=True
    )
    parser.add_argument(
        '-s',
        '--shuffle',
        help='shuffle mode (bool)',
        default=True,
        action='store_true'
    )
    parser.add_argument(
        '-f1',
        '--output_file1',
        help='output file name 1 (csv) default=training_for_centernet.csv',
        default='training_for_centernet.csv',
        required=True
    )
    parser.add_argument(
        '-r1',
        '--ratio1',
        help='output file ratio 1 (int) default=90',
        default=90,
        type=int,
        required=True
    )
    parser.add_argument(
        '-f2',
        '--output_file2',
        help='output file name 2 (csv) default=validation_for_centernet.csv',
        default='validation_for_centernet.csv'
    )
    parser.add_argument(
        '-r2',
        '--ratio2',
        help='output file ratio 2 (int) default=10',
        default=10,
        type=int
    )

    args = parser.parse_args()

    # Set args
    dataset_dict = dict()
    dataset_dict["inputdir"] = args.input_dir
    dataset_dict["outdir"] = args.output_dir
    dataset_dict["num_class"] = args.num_class
    dataset_dict["channel"] = args.channel
    dataset_dict["width"], dataset_dict["height"] = args.width, args.height
    dataset_dict["grid_size"] = args.grid_size
    dataset_dict["mode"] = args.mode
    dataset_dict["shuffle"] = args.shuffle
    dataset_dict["file1"] = args.output_file1
    dataset_dict["ratio1"] = args.ratio1
    dataset_dict["file2"] = args.output_file2
    dataset_dict["ratio2"] = args.ratio2

    object_detection_centernet_util.create_object_detection_dataset_command(
        dataset_dict)

    logger.log(99, 'Dataset creation completed successfully.')


if __name__ == '__main__':
    main()
