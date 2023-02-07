# Copyright 2023 Sony Group Corporation.
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
from nnabla.utils.image_utils import imsave
from abn_utils.abn import load_model_config, abn_attention_map_single
from abn_utils.args import get_single_image_args


def func(args):
    config = load_model_config(args)
    result = abn_attention_map_single(args.image, config)
    imsave(args.output, result, channel_first=True)


def main():
    parser = argparse.ArgumentParser(
        description='Attention Map Visualization\n' +
        'Attention Branch Network: Learning of Attention Mechanism for Visual Explanation\n' +
        'Hiroshi Fukui, Tsubasa Hirakawa, Takayoshi Yamashita, Hironobu Fujiyoshi\n' +
        'https://ieeexplore.ieee.org/document/8953929\n' +
        '', formatter_class=argparse.RawTextHelpFormatter)

    args = get_single_image_args(parser)
    func(args)
    logger.log(99, 'ABN (attention map) completed successfully.')


if __name__ == '__main__':
    main()
