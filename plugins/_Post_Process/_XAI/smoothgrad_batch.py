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
from smoothgrad_utils.args import get_multi_image_args
from utils.file import save_info_to_csv
from smoothgrad_utils.smoothgrad_func import smoothgrad_batch_func


def main():
    parser = argparse.ArgumentParser(
        description='SmoothGrad(batch)\n' +
        '\n' +
        'SmoothGrad: removing noise by adding noise\n' +
        'Daniel Smilkov, Nikhil Thorat, Been Kim, Fernanda Viegas, Martin Wattenberg\n' +
        'https://arxiv.org/abs/1706.03825\n' +
        '', formatter_class=argparse.RawTextHelpFormatter)
    args = get_multi_image_args(parser)

    file_names = smoothgrad_batch_func(args)

    save_info_to_csv(args.input, args.output, file_names, 'SmoothGrad')
    logger.log(99, 'SmoothGrad completed successfully.')


if __name__ == '__main__':
    main()
