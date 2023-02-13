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
import functools
import numpy as np
from nnabla.utils.cli.utility import let_data_to_variable
from utils.image import preprocess_image, get_gray_image
from utils.model import load_executor, zero_grad
from utils.model import get_num_classes, is_binary_classification
from utils.layer import get_layer_name_from_idx


def smoothgrad(
    executor, _let_d_to_v, im, target_layer, output_variable,
    class_index, noise_level, num_samples, magnitude=True
):
    selected = output_variable.variable_instance[:, class_index]
    selected.need_grad = True
    stdev = noise_level * (np.max(im) - np.min(im))
    accum_grad = np.zeros(
        target_layer.variable_instance.shape[1:], dtype=np.float32)
    for _ in range(num_samples):
        noise = np.random.normal(0, stdev, im.shape)
        im_with_noise = im + noise
        _let_d_to_v(data=im_with_noise)
        # forward, backward
        zero_grad(executor)
        selected.grad.zero()
        selected.forward()
        selected.backward()

        # add grad
        grad = target_layer.variable_instance.g.copy()[0]
        if magnitude:
            accum_grad += (grad * grad)
        else:
            accum_grad += grad
    # averaged grad
    return accum_grad / num_samples


def get_smoothgrad_image(input_image, config):
    executor = config.executor
    output_variable = config.output_variable
    input_variable = config.input_variable
    # input image
    im = preprocess_image(input_image, executor.no_image_normalization)
    _let_d_to_v = functools.partial(
        let_data_to_variable,
        variable=input_variable.variable_instance,
        data_name=config.input_data_name,
        variable_name=input_variable.name
    )
    layer_name = get_layer_name_from_idx(
        executor.network.variables, config.layer_index)
    target_layer = executor.network.variables[layer_name]

    sg = smoothgrad(
        executor, _let_d_to_v, im,
        target_layer, output_variable,
        config.class_index, config.noise_level, config.num_samples
    )
    # Generate output image
    ret = get_gray_image(sg)
    return ret


def get_config(args):
    class Config:
        pass
    config = Config()
    executor = load_executor(args.model)

    # Prepare variables
    input_variable, data_name = list(executor.dataset_assign.items())[0]
    config.input_variable = input_variable
    config.input_data_name = data_name
    config.output_variable = list(executor.output_assign.keys())[0]
    config.input_shape = input_variable.shape[-2:]
    config.executor = executor
    # multi-image version
    if hasattr(args, 'input'):
        # judge binary or multi class
        num_classes = get_num_classes(args.input, args.label_variable)
        config.is_binary_clsf = is_binary_classification(
            num_classes, config.output_variable.variable_instance.d.shape[1]
        )
        config.class_index = None
    # single-image version
    else:
        config.class_index = args.class_index

    config.num_samples = args.num_samples
    config.noise_level = args.noise_level
    config.layer_index = args.layer_index
    return config
