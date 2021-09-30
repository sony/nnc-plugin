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
import tqdm
from nnabla import logger
from nnabla.utils.image_utils import imsave
from nnabla.utils.data_iterator import data_iterator_csv_dataset
from gradcam_utils.gradcam import get_gradcam_image
from gradcam_utils.utils import save_info_to_csv, get_class_index
from gradcam_utils.setup_model import get_config
from utils.file import save_info_to_csv


def func(args):
    config = get_config(args)
    # Load dataset
    data_iterator = (lambda: data_iterator_csv_dataset(
        uri=args.input,
        batch_size=1,
        shuffle=False,
        normalize=not config.executor.no_image_normalization,
        with_memory_cache=False,
        with_file_cache=False))

    # Prepare output
    data_output_dir = os.path.splitext(args.output)[0]

    # Data loop
    with data_iterator() as di:
        pbar = tqdm.tqdm(total=di.size)
        index = 0
        file_names = []
        while index < di.size:
            # Load data
            data = di.next()
            im = data[di.variables.index(args.input_variable)]
            label = data[di.variables.index(args.label_variable)]
            label = label.reshape((label.size,))
            config.class_index = get_class_index(label, config.is_binary_clsf)
            result = get_gradcam_image(im[0], config)

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

    save_info_to_csv(args.input, args.output,
                     file_names, column_name='gradcam')
    logger.log(99, 'Grad-CAM completed successfully.')


def main():
    parser = argparse.ArgumentParser(
        description='Grad-CAM (batch)\n' +
        '\n' +
        'Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization\n' +
        'Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, Dhruv Batra\n' +
        'https://arxiv.org/abs/1610.02391\n' +
        '', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-i', '--input', help='path to input csv file (csv) default=output_result.csv', required=True, default='output_result.csv')
    parser.add_argument(
        '-m', '--model', help='path to model nnp file (model) default=results.nnp', required=True, default='results.nnp')
    parser.add_argument(
        '-v1', '--input_variable', help='Variable to be processed (variable), default=x', required=True, default='x')
    parser.add_argument(
        '-v2', '--label_variable', help='Variable representing class index to visualize (variable) default=y', required=True, default='y')
    parser.add_argument(
        '-o', '--output', help='path to output csv file (csv) default=gradcam.csv', required=True, default='gradcam.csv')
    # designage if the model contains crop between input and first conv layer.
    parser.add_argument(
        '-cr', '--contains_crop', help=argparse.SUPPRESS, action='store_true')
    parser.set_defaults(func=func)

    args = parser.parse_args()

    args.func(args)


if __name__ == '__main__':
    main()
