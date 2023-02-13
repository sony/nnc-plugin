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


import os
from tqdm import tqdm
import nnabla as nn
import nnabla.utils.load as load
from nnabla.utils.load import _create_dataset
from nnabla.utils.image_utils import imresize, imread, imsave
from nnabla.utils.cli.utility import let_data_to_variable
from utils.image import generate_heatmap, normalize_image, overlay_images, preprocess_image


def abn_attention_map_single(image_path, config):
    executor = list(config.executors)[-1]
    net = executor.network
    inputs = {i: net.variables[i].variable_instance for i in net.inputs}
    outputs = {i: net.variables[i].variable_instance for i in net.outputs}
    input_image = list(inputs.values())[0]
    input_name = list(inputs.keys())[0]
    att_map = net.variables[config.attention_map_layer]
    input_shape = input_image[0].shape[-2:]
    _h, _w = input_shape
    image = imread(image_path, size=(_w, _h), channel_first=True)
    image = preprocess_image(image, executor.no_image_normalization)
    let_data_to_variable(
        input_image,
        image,
        data_name=input_name,
        variable_name=list(executor.dataset_assign.values())[0]
    )
    losses = list(outputs.values())
    nn.forward_all(losses)
    ret = _get_attention_map(input_image, att_map, input_shape)
    return ret


def abn_attention_map_batch(config):
    # Prepare output
    data_output_dir = config.data_output_dir
    executor = list(config.executors)[-1]
    from collections import OrderedDict
    datasets = OrderedDict()
    i = 1
    d = config.proto.dataset[i]
    datasets[d.name] = _create_dataset(
        config.input,
        config.batch_size,
        d.shuffle,
        d.no_image_normalization,
        d.cache_dir,
        d.overwrite_cache,
        d.create_cache_explicitly,
        True,
        i)

    executor.data_iterators = OrderedDict()
    for d in datasets.keys():
        executor.data_iterators[d] = datasets[d].data_iterator
    # Data loop
    net = executor.network
    inputs = {i: net.variables[i].variable_instance for i in net.inputs}
    outputs = {i: net.variables[i].variable_instance for i in net.outputs}
    input_image = list(inputs.values())[0]
    att_map = net.variables[config.attention_map_layer]
    input_shape = list(inputs.values())[0].shape[-2:]
    losses = list(outputs.values())
    with list(executor.data_iterators.values())[0]() as di:
        pbar = tqdm(total=di.size)
        index = 0
        file_names = []
        while index < di.size:
            image, _ = di.next()
            input_image.d = image
            nn.forward_all(losses, clear_no_need_grad=True)
            result = _get_attention_map(input_image, att_map, input_shape)
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

    return file_names


def load_model_config(args, batch_size=1):
    class ForwardConfig:
        pass
    info = load.load(args.model, prepare_data_iterator=False,
                     batch_size=batch_size)

    config = ForwardConfig
    config.global_config = info.global_config
    config.training_config = info.training_config
    config.optimizers = info.optimizers.values()
    config.monitors = info.monitors.values()
    config.executors = info.executors.values()
    config.datasets = info.datasets
    config.proto = info.proto
    config.batch_size = batch_size
    config.data_output_dir = os.path.splitext(args.output)[0]
    config.attention_map_layer = args.attention_map_layer
    if hasattr(args, 'input'):
        config.input = args.input
    return config


def get_attention_map(input_image, attention_layer, input_shape):
    _layer = normalize_image(attention_layer)
    _h, _w = input_shape
    heatmap = generate_heatmap(_layer)
    heatmap = imresize(heatmap, (_w, _h), channel_first=True)
    result = overlay_images(heatmap, input_image)
    return result


def _get_attention_map(input_image, att_map, input_shape):
    return get_attention_map(input_image.d[0], att_map.variable_instance.d, input_shape)
