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
import numpy as np
from nnabla import logger
from nnabla.ext_utils import get_extension_context
from nnabla.utils.load import load


def get_num_classes(input_path, label_variable):
    csv_file = np.loadtxt(input_path, delimiter=",", dtype=str)
    header = [item.split(":")[0] for item in csv_file[0]]
    classes = csv_file[1:, header.index(label_variable)]
    num_classes = np.unique(classes).size
    return num_classes


def is_binary_classification(num_classes, output_classes):
    return num_classes != output_classes


def get_class_index(label, is_binary_clsf):
    if is_binary_clsf:
        class_index = 0
    else:
        class_index = np.argmax(label)
    return int(class_index)


def get_context(device_id):
    # for cli app use
    try:
        context = 'cudnn'
        ctx = get_extension_context(context, device_id=device_id)
    except (ModuleNotFoundError, ImportError):
        context = 'cpu'
        ctx = get_extension_context(context, device_id=device_id)
    # for nnc use
    config_filename = 'net.nntxt'
    if os.path.isfile(config_filename):
        config_info = load([config_filename])
        ctx = config_info.global_config.default_context
    return ctx


def zero_grad(executor):
    for k, v in executor.network.variables.items():
        v.variable_instance.need_grad = True
        v.variable_instance.grad.zero()


def load_executor(nnp_file):
    class ForwardConfig:
        pass

    # Load model
    info = load([nnp_file], prepare_data_iterator=False, batch_size=1)

    config = ForwardConfig
    config.global_config = info.global_config

    config.executors = info.executors.values()

    config.networks = []
    if len(config.executors) < 1:
        logger.critical('Executor is not found in {}.'.format(nnp_file))
        return
    executor = list(config.executors)[0]
    if len(config.executors) > 1:
        logger.log(99, 'Only the first executor {} is used in the plugin calculation.'.format(
            executor.name))

    if executor.network.name in info.networks.keys():
        config.networks.append(info.networks[executor.network.name])
    else:
        logger.critical('Network {} is not found in {}.'.format(
            executor.network.name, nnp_file))
        return

    if len(executor.dataset_assign.items()) > 1:
        logger.critical('This plugin only supports single image input model.')
    return executor
