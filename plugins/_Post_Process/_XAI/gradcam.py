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
from gradcam.gradcam import get_gradcam_image
from gradcam.setup_model import get_config


def func(args):
    config = get_config(args)
    _h, _w = config.input_shape
    input_image = imread(args.image, size=(_w, _h), channel_first=True)
    result = get_gradcam_image(input_image, config)
    imsave(args.output, result, channel_first=True)
    logger.log(99, 'Grad-CAM completed successfully.')


def main():
    parser = argparse.ArgumentParser(
        description='Grad-CAM\n' +
        '\n' +
        'Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization\n' +
        'Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, Dhruv Batra\n' +
        'https://arxiv.org/abs/1610.02391\n' +
        '', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-m', '--model', help='path to model nnp file (model) default=results.nnp', required=True, default='results.nnp')
    parser.add_argument(
        '-i', '--image', help='path to input image file (image)', required=True)
    parser.add_argument(
        '-c', '--class_index', help='class index to visualize (int), default=0', required=True, type=int, default=0)
    parser.add_argument(
        '-o', '--output', help='path to output image file (image) default=grad_cam.png', required=True, default='grad_cam.png')
    # designage if the model contains crop between input and first conv layer.
    parser.add_argument(
        '-cr', '--contains_crop', help=argparse.SUPPRESS, action='store_true')
    parser.set_defaults(func=func)

    args = parser.parse_args()

    args.func(args)


if __name__ == '__main__':
    main()
