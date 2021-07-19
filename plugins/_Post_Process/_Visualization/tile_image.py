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
import csv
import numpy as np
from nnabla import logger
from nnabla.utils.image_utils import imread, imresize, imsave


def func(args):
    with open(args.input, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        table = [row for row in reader]

    num_data = len(table)
    if args.end_index < 0:
        args.end_index = num_data - 1
    elif args.end_index >= len(table):
        args.end_index = len(table) - 1
    table = table[args.start_index:args.end_index + 1]

    if args.variable is None:
        args.variable = ','.join(
            [col.split('__')[0].split(':')[0] for col in header])
    col_names = list(set(args.variable.split(',')))
    cols = []
    for col_name in col_names:
        cols.extend([j for j, col in enumerate(header)
                     if col_name == col.split('__')[0].split(':')[0]])

    images = []
    for row in table:
        for col in cols:
            if '.png' in row[col] or '.jpg' in row[col] or '.jpeg' in row[col] or '.tif' in row[col] or '.bmp' in row[col]:
                images.append(row[col])
    if len(images) == 0:
        logger.critical(
            99, 'Image file is not found in variable {}.'.format(
                args.variable))
        return

    logger.log(
        99,
        'Tile {} images included in {} lines.'.format(
            len(images),
            len(table)))

    if args.image_width < 0 or args.image_height < 0:
        im = imread(images[0])
        if args.image_width < 0:
            args.image_width = im.shape[1]
        if args.image_height < 0:
            args.image_height = im.shape[0]
        if args.image_width * args.num_column > 3840:
            scale = 3840 / (args.image_width * args.num_column)
            args.image_width = int(args.image_width * scale)
            args.image_height = int(args.image_height * scale)
    logger.log(
        99,
        'Image width = {}, Image height = {}.'.format(
            args.image_width,
            args.image_height))

    result = np.ndarray((args.image_height *
                         (1 +
                          (len(images) -
                           1) //
                             args.num_column), args.image_width *
                         args.num_column, 3), dtype=np.uint8)
    result.fill(0)
    for i, image in enumerate(images):
        col = i % args.num_column
        row = i // args.num_column
        im = imread(image)
        if im.shape[1] != args.image_width or im.shape[0] != args.image_height:
            im = imresize(im, (args.image_width, args.image_height))
        if len(im.shape) < 3:
            im = im.reshape(im.shape + (1,))
        result[row *
               args.image_height:(row +
                                  1) *
               args.image_height, col *
               args.image_width:(col +
                                 1) *
               args.image_width] = im

    imsave(args.output, result)

    logger.log(99, 'Tile images completed successfully.')


def main():
    parser = argparse.ArgumentParser(
        description='Tile Images\n\nTile the images in the input dataset CSV file\n\n',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-i',
        '--input',
        help='path to input csv file (csv) default=output_result.csv',
        required=True,
        default='output_result.csv')
    parser.add_argument(
        '-v',
        '--variable',
        help="image variables to tile. separate by comma if more than one (variables)")
    parser.add_argument(
        '-w',
        '--image_width',
        help='width of each image. automatic if not specified (int)',
        type=int,
        default=-1)
    parser.add_argument(
        '-g',
        '--image_height',
        help='height of each image. automatic if not specified (int)',
        type=int,
        default=-1)
    parser.add_argument(
        '-c',
        '--num_column',
        help="number of image arranged in a row (int) default=32",
        default=32,
        type=int)
    parser.add_argument(
        '-s',
        '--start_index',
        help='index of first data (int). default=0',
        default=0,
        type=int)
    parser.add_argument(
        '-e',
        '--end_index',
        help='index of last data. use all images if not specified (int), default=1023',
        default=-
        1,
        type=int)
    parser.add_argument(
        '-o',
        '--output',
        help='path to output image file (image) default=tiled_images.png',
        required=True,
        default='tiled_images.png')
    parser.set_defaults(func=func)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
