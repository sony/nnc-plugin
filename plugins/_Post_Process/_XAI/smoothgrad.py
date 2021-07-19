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
from nnabla import logger
from nnabla.utils.image_utils import imsave, imread
from smoothgrad.smoothgrad import get_smoothgrad_image, get_config
from smoothgrad.args import get_single_image_args


def func(args):
    config = get_config(args)
    _h, _w = config.input_shape
    input_image = imread(args.image, size=(_w, _h), channel_first=True)
    result = get_smoothgrad_image(input_image, config)
    imsave(args.output, result, channel_first=True)
    logger.log(99, 'SmoothGrad completed successfully.')


def main():
    parser = argparse.ArgumentParser(
        description='SmoothGrad\n' +
        '\n' +
        'SmoothGrad: removing noise by adding noise\n' +
        'Daniel Smilkov, Nikhil Thorat, Been Kim, Fernanda Viegas, Martin Wattenberg\n' +
        'https://arxiv.org/abs/1706.03825\n' +
        '', formatter_class=argparse.RawTextHelpFormatter)
    args = get_single_image_args(parser)
    func(args)


if __name__ == '__main__':
    main()
