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
from nnabla import logger
import nnabla.utils.load as load
from .utils import get_layer_shape, get_last_conv_name, get_first_conv_name
from .utils import is_binary_classification, get_num_classes


def load_executor(nnp_file):
    class ForwardConfig:
        pass

    # Load model
    info = load.load([nnp_file], prepare_data_iterator=False, batch_size=1)

    config = ForwardConfig
    config.global_config = info.global_config

    config.executors = info.executors.values()

    config.networks = []
    if len(config.executors) < 1:
        logger.critical('Executor is not found in {}.'.format(nnp_file))
        return
    executor = list(config.executors)[0]
    if len(config.executors) > 1:
        logger.log(
            99, 'Only the first executor {} is used in the Grad-CAM calculation.'.format(executor.name))

    if executor.network.name in info.networks.keys():
        config.networks.append(info.networks[executor.network.name])
    else:
        logger.critical('Network {} is not found in {}.'.format(
            executor.network.name, nnp_file))
        return

    if len(executor.dataset_assign.items()) > 1:
        logger.critical(
            'Grad-CAM plugin only supports single image input model.')
    return executor


def get_config(args):
    class GradcamConfig:
        pass
    config = GradcamConfig()
    executor = load_executor(args.model)
    last_conv_name = get_last_conv_name(executor.network.variables)
    if last_conv_name is None:
        logger.critical('Convolution is not found in the network {}.'.format(
            executor.network.name))
        raise RuntimeError(
            "At least one convolution layer is necessary in the model to use this plugin.")
    logger.log(99, "Using Grad at {}".format(last_conv_name))

    # Prepare variables
    input_variable, data_name = list(executor.dataset_assign.items())[0]
    output_variable = list(executor.output_assign.keys())[0]
    input_shape = input_variable.shape[-2:]

    # set padding if model contains crop
    if args.contains_crop:
        first_conv_name = get_first_conv_name(executor.network.variables)
        cropped_shape = get_layer_shape(
            executor.network.variables, first_conv_name)[-2:]
        config.cropped_shape = cropped_shape
        pad_h = int((input_shape[0] - cropped_shape[0]) / 2)
        pad_w = int((input_shape[1] - cropped_shape[1]) / 2)
        config.padding = (pad_h, pad_w)
    else:
        config.cropped_shape = input_shape
        config.padding = (0, 0)

    # multi-image version
    if hasattr(args, 'input'):
        # judge binary or multi class
        num_classes = get_num_classes(args.input, args.label_variable)
        config.is_binary_clsf = is_binary_classification(
            num_classes, output_variable.variable_instance.d.shape[1]
        )
        config.class_index = None
    # single-image version
    else:
        config.class_index = args.class_index

    config.executor = executor
    config.input_shape = input_shape
    config.input_data_name = data_name
    config.input_variable = input_variable
    config.output_variable = output_variable
    config.last_conv_name = last_conv_name
    return config
