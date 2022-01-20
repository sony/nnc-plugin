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
from tqdm import tqdm
from nnabla import logger
from nnabla.utils.image_utils import imsave
from nnabla.utils.data_iterator import data_iterator_csv_dataset
from smoothgrad_utils.smoothgrad import get_smoothgrad_image, get_config
from smoothgrad_utils.args import get_multi_image_args
from utils.file import save_info_to_csv
from utils.model import get_class_index


def func(args):
    config = get_config(args)
    # Load dataset
    data_iterator = data_iterator_csv_dataset(
        uri=args.input,
        batch_size=1,
        shuffle=False,
        normalize=not config.executor.no_image_normalization,
        with_memory_cache=False,
        with_file_cache=False
    )

    # Prepare output
    data_output_dir = os.path.splitext(args.output)[0]

    # Data loop
    with data_iterator as di:
        pbar = tqdm(total=di.size)
        index = 0
        file_names = []
        while index < di.size:
            # Load data
            data = di.next()
            im = data[di.variables.index(args.input_variable)]
            label = data[di.variables.index(args.label_variable)]
            label = label.reshape((label.size,))
            config.class_index = get_class_index(label, config.is_binary_clsf)
            result = get_smoothgrad_image(im[0], config)
            # Output result image
            file_name = os.path.join(
                data_output_dir,
                '{:04d}'.format(index // 1000),
                '{}.png'.format(index)
            )
            directory = os.path.dirname(file_name)
            try:
                os.makedirs(directory)
            except OSError:
                pass  # python2 does not support exists_ok arg
            imsave(file_name, result, channel_first=True)

            file_names.append(file_name)
            index += 1
            pbar.update(1)
        pbar.close()

    save_info_to_csv(args.input, args.output, file_names, 'SmoothGrad')
    logger.log(99, 'SmoothGrad completed successfully.')


def main():
    parser = argparse.ArgumentParser(
        description='SmoothGrad(batch)\n' +
        '\n' +
        'SmoothGrad: removing noise by adding noise\n' +
        'Daniel Smilkov, Nikhil Thorat, Been Kim, Fernanda Viegas, Martin Wattenberg\n' +
        'Workshop on Visualization for Deep Learning, ICML, 2017.\n' +
        'https://arxiv.org/abs/1706.03825\n' +
        '', formatter_class=argparse.RawTextHelpFormatter)
    args = get_multi_image_args(parser)
    func(args)


if __name__ == '__main__':
    main()
