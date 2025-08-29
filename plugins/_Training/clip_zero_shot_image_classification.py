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
import os
import sys
import argparse

import numpy as np
import nnabla as nn
import nnabla.functions as F
from nnabla import logger
from nnabla.ext_utils import get_extension_context
from nnabla.utils.download import download
from nnabla.models.utils import get_model_home, get_model_url_base
from nnabla.utils.save import save

# autopep8: off
sys.path.append(os.path.join(os.path.dirname(__file__),
                '..', '_NNablaExamples', 'nnabla-examples'))
import vision_and_language.clip.clip as clip
# autopep8: on


def func(args):
    # check options
    if args.query_text is not None and args.query_text_csv_file is not None:
        logger.critical(
            'Specify only one of the two options query_text and query_text_csv_file.')
        exit(1)
    if args.query_text is not None:
        query_text = args.query_text.split(',')
    elif args.query_text_csv_file is not None:
        with open(args.query_text_csv_file, 'r') as f:
            query_text = f.readlines()
    else:
        logger.critical('query_text or query_text_csv_file not specified.')
        exit(1)

    # load model
    if args.model.lower() == 'vitb16':
        model_name = 'ViT-B-16'
    elif args.model.lower() == 'vitb32':
        model_name = 'ViT-B-32'
    elif args.model.lower() == 'vitl14':
        model_name = 'ViT-L-14'
    else:
        logger.critical(f'{args.model} is not supported.')
        sys.exit(1)
    path_nnp = os.path.join(
        get_model_home(), 'clip', f'{model_name}.h5')
    url = f'https://zenodo.org/records/16973930/files/{model_name}.h5?download=1'
    logger.log(99, f'Downloading {model_name} from {url}...')
    dir_nnp = os.path.dirname(path_nnp)
    if not os.path.isdir(dir_nnp):
        os.makedirs(dir_nnp)
    download(url, path_nnp, open_file=False, allow_overwrite=False)

    logger.log(99, f'Loading {model_name}...')
    clip.load(path_nnp)

    params = nn.get_parameters()
    vision_patch_size = params["visual/conv1/W"].shape[-1]

    logger.log(99, f'Preparing model...')
    # prepare test text feature
    text = nn.Variable.from_numpy_array(clip.tokenize(query_text))

    with nn.auto_forward():
        text_features = clip.encode_text(text)

    # prepare model
    x = nn.Variable((1, 3, 224, 224))

    mean = nn.parameter.get_parameter_or_create(
        name="mean",
        shape=(1, 3, 1, 1),
        initializer=(np.asarray(
            [0.48145466, 0.4578275, 0.40821073]) * 255.0).reshape(1, 3, 1, 1),
        need_grad=False)
    std = nn.parameter.get_parameter_or_create(
        name="std",
        shape=(1, 3, 1, 1),
        initializer=(np.asarray(
            [0.26862954, 0.26130258, 0.27577711]) * 255.0).reshape(1, 3, 1, 1),
        need_grad=False)

    normalized_x = (x - mean) / std

    image_features = clip.encode_image(normalized_x)

    text_features_param = nn.parameter.get_parameter_or_create(
        name="text_feature",
        shape=text_features.shape,
        initializer=text_features.d,
        need_grad=False)

    # normalized features
    image_features = image_features / \
        F.norm(image_features, axis=1, keepdims=True)
    text_features = text_features_param / \
        F.norm(text_features_param, axis=1, keepdims=True)

    # cosine similarity as logits
    logit_scale = nn.parameter.get_parameter_or_create(
        name='logit_scale', shape=())
    logit_scale = F.exp(logit_scale)

    image_features = image_features.reshape(
        (1, image_features.shape[0], image_features.shape[1]))
    text_features = F.transpose(text_features, (1, 0))
    text_features = text_features.reshape(
        (1, text_features.shape[0], text_features.shape[1]))

    per_image = F.batch_matmul(image_features, text_features).reshape(
        (image_features.shape[0], -1))
    logits_per_image = logit_scale.reshape((1, 1)) * per_image

    probs = F.softmax(logits_per_image, axis=-1)

    # save model to nnp file
    logger.log(99, f'Saving model...')
    contents = {
        'global_config': {'default_context': get_extension_context('cudnn')},
        'networks': [
            {'name': 'runtime',
             'batch_size': 1,
             'outputs': {'y\'': probs},
             'names': {'x': x}}],
        'executors': [
            {'name': 'runtime',
             'network': 'runtime',
             'no_image_normalization': True,
             'data': ['x'],
             'output': ['y\'']}]}
    save(os.path.join(args.output_dir, 'results.nnp'),
         contents, variable_batch_size=False)
    logger.log(99, 'Training Completed.')


def main():
    parser = argparse.ArgumentParser(
        description='Zero-shot image classification(CLIP)\n' +
        '\n' +
        'Create an image classifier from the query text for each class by using CLIP.\n' +
        'Set query text for each image class by either qyery_text of query_text_csv_file.\n' +
        'This plug-in works with RGB image with 224 x 224 pixels.\n' +
        '\n' +
        'Learning Transferable Visual Models From Natural Language Supervision\n' +
        'Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever\n' +
        'https://arxiv.org/abs/2103.00020', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-c',
        '--config',
        help='config file(nntxt) default=net.nntxt',
        required=True,
        default='net.nntxt')
    parser.add_argument(
        '-o',
        '--output_dir',
        help='output_dir(dir)',
        required=True)
    parser.add_argument(
        '-m',
        '--model',
        help='model(option:vitb16,vitb32,vitl14),default=vitb32',
        default='vitb32', required=True)
    parser.add_argument(
        '-q',
        '--query_text',
        help='comma separated query text for each image class(text)')
    parser.add_argument(
        '-qc',
        '--query_text_csv_file',
        help='A text file with query text for each image class on each line(file)')
    parser.set_defaults(func=func)

    args = parser.parse_args()

    args.func(args)


if __name__ == '__main__':
    main()
