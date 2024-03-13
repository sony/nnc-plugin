# Copyright 2024 Sony Group Corporation.
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
from fileinput import filename
import os
import argparse
import numpy as np
import nnabla as nn
from nnabla import logger
from nnabla.utils.image_utils import imread
from nnabla.models.semantic_segmentation import DeepLabV3plus
from nnabla.models.semantic_segmentation.utils import ProcessImage


def func(args):
    logger.log(
        99, f'Preparing DeepLabv3plus dataset={args.dataset} output_stride={args.output_stride}...')
    # Get context
    from nnabla.ext_utils import get_extension_context
    nn.set_default_context(get_extension_context('cudnn', device_id='0'))

    # Build a Deeplab v3+ network
    image = imread(args.input)
    x = nn.Variable((1, 3, args.target_h, args.target_w), need_grad=False)
    deeplabv3 = DeepLabV3plus(args.dataset, output_stride=args.output_stride)
    y = deeplabv3(x)

    # preprocess image
    processed_image = ProcessImage(image, args.target_h, args.target_w)
    input_array = processed_image.pre_process()

    # Compute inference
    logger.log(99, 'Processing ...')
    x.d = input_array
    y.forward(clear_buffer=True)
    output = np.argmax(y.d, axis=1)

    # Apply post processing
    post_processed = processed_image.post_process(output[0])

    # Display predicted class names
    predicted_classes = np.unique(post_processed).astype(int)
    for i in range(predicted_classes.shape[0]):
        print('Classes Segmented: ',
              deeplabv3.category_names[predicted_classes[i]])

    # save inference result
    processed_image.save_segmentation_image(args.output)

    logger.log(99, 'Plugin completed successfully.')


def main():
    parser = argparse.ArgumentParser(
        description='DeepLabv3plus\n\n' +
        'Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation\n' +
        'Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, Hartwig Adam\n' +
        'https://arxiv.org/abs/1802.02611v3\n\n',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-i',
        '--input',
        help='input image(image)',
        required=True)
    parser.add_argument(
        '-tw',
        '--target_w',
        help='target width(int), default=513',
        type=int,
        default=513)
    parser.add_argument(
        '-th',
        '--target_h',
        help='target height(int), default=513',
        type=int,
        default=513)
    parser.add_argument(
        '-d',
        '--dataset',
        help='training dataset name(option:voc,voc-coco), default=voc-coco',
        default='voc-coco')
    parser.add_argument(
        '-s',
        '--output_stride',
        help='output stride(option:8,16), default=8',
        type=int,
        default=8)
    parser.add_argument(
        '-o',
        '--output',
        help='path to output png file(file), default=deeplabv3plus_output.png',
        required=True)
    parser.set_defaults(func=func)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
