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
from nnabla.utils.cli.utility import let_data_to_variable
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
import nnabla as nn
from nnabla import logger
import nnabla.utils.load as load
import matplotlib
matplotlib.use('Agg')


def get_executor(model):
    class ForwardConfig:
        pass
    # Load model
    info = load.load([model], prepare_data_iterator=False, batch_size=1)

    config = ForwardConfig
    config.global_config = info.global_config

    config.executors = info.executors.values()

    config.networks = []
    if len(config.executors) < 1:
        logger.critical('Executor is not found in {}.'.format(model))
        return
    executor = list(config.executors)[0]
    if len(config.executors) > 1:
        logger.log(99, 'Only the first executor {} is used in the SHAP calculation.'.format(
            executor.name))

    if executor.network.name in info.networks.keys():
        config.networks.append(info.networks[executor.network.name])
    else:
        logger.critical('Network {} is not found in {}.'.format(
            executor.network.name, model))
        return
    return executor


def red_blue_map():
    colors = []
    for i in np.linspace(1, 0, 100):
        colors.append((30. / 255, 136. / 255, 229. / 255, i))
    for i in np.linspace(0, 1, 100):
        colors.append((255. / 255, 13. / 255, 87. / 255, i))
    return LinearSegmentedColormap.from_list("red_transparent_blue", colors)


def gradient(model, idx, inputs, batch_size, interim_layer):

    class ForwardConfig:
        pass
    # Load model
    info = load.load([model], prepare_data_iterator=False,
                     batch_size=batch_size)

    config = ForwardConfig
    config.global_config = info.global_config

    config.executors = info.executors.values()

    config.networks = []
    if len(config.executors) < 1:
        logger.critical('Executor is not found in {}.'.format(model))
        return
    executor = list(config.executors)[0]
    if len(config.executors) > 1:
        logger.log(99, 'Only the first executor {} is used in the SHAP calculation.'.format(
            executor.name))

    if executor.network.name in info.networks.keys():
        config.networks.append(info.networks[executor.network.name])
    else:
        logger.critical('Network {} is not found in {}.'.format(
            executor.network.name, model))
        return

    # Prepare variable
    input_variable, data_name = list(executor.dataset_assign.items())[0]
    output_variable = list(executor.output_assign.keys())[0]

    for k, v in executor.network.variables.items():
        v.variable_instance.need_grad = True
        v.variable_instance.grad.zero()

    # input image
    let_data_to_variable(input_variable.variable_instance,
                         np.reshape(
                             inputs, input_variable.variable_instance.d.shape),
                         data_name=data_name, variable_name=input_variable.name)
    input_variable.variable_instance.need_grad = True

    # Forward
    output_variable.variable_instance.forward()
    with nn.auto_forward():
        selected = output_variable.variable_instance[:, int(idx)]

    # Generate data
    for v, generator in executor.generator_assign.items():
        v.variable_instance.d = generator(v.shape)

    # Backward
    selected.backward()

    if interim_layer == 0:
        grads = [input_variable.variable_instance.g]

    else:
        layers = dict()
        for k, v in executor.network.variables.items():
            layers[k] = v.variable_instance
        for i, k in enumerate(layers):
            if i + 1 == int(interim_layer):
                grads = [layers[k].g.copy()]
            else:
                continue
    return grads


def get_interim_input(model, inputs, interim_layer):

    if len(inputs.shape) == 3:
        batch_size = 1
    else:
        batch_size = len(inputs)

    class ForwardConfig:
        pass
    # Load model
    info = load.load([model], prepare_data_iterator=False,
                     batch_size=batch_size)

    config = ForwardConfig
    config.global_config = info.global_config

    config.executors = info.executors.values()

    config.networks = []
    if len(config.executors) < 1:
        logger.critical('Executor is not found in {}.'.format(model))
        return
    executor = list(config.executors)[0]
    if len(config.executors) > 1:
        logger.log(99, 'Only the first executor {} is used in the SHAP calculation.'.format(
            executor.name))

    if executor.network.name in info.networks.keys():
        config.networks.append(info.networks[executor.network.name])
    else:
        logger.critical('Network {} is not found in {}.'.format(
            executor.network.name, model))
        return

    # Prepare variable
    input_variable, data_name = list(executor.dataset_assign.items())[0]
    output_variable = list(executor.output_assign.keys())[0]

    for k, v in executor.network.variables.items():
        v.variable_instance.grad.zero()

    # input image
    let_data_to_variable(input_variable.variable_instance,
                         np.reshape(
                             inputs, input_variable.variable_instance.d.shape),
                         data_name=data_name, variable_name=input_variable.name)

    # Forward
    output_variable.variable_instance.forward()

    # Generate data
    for v, generator in executor.generator_assign.items():
        v.variable_instance.d = generator(v.shape)

    layers = dict()
    for k, v in executor.network.variables.items():
        layers[k] = v.variable_instance
    if int(interim_layer) > len(layers) or int(interim_layer) < 1:
        return None
    for i, k in enumerate(layers):
        if i + 1 == int(interim_layer):
            interim_layer = layers[k].d.copy()
            return interim_layer
        else:
            continue


def plot_shap(model, X, label, output, interim_layer, num_samples, data_iterator,
              batch_size, red_blue_map, gradient, get_interim_input):
    output_phis = []
    if interim_layer == 0:
        data = X.reshape((1,) + X.shape)
    else:
        data = get_interim_input(model, X, interim_layer)
        if data is None:
            logger.log(
                99, 'The interim layer should be an integer between 1 and the number of layers of the model!')
            return
        if len(data.shape) != 4:
            logger.log(
                99, 'The input of the interim layer must have the shape of (samples x channels x width x height)')
            return

    samples_input = [np.zeros((num_samples, ) + X.shape)]
    samples_delta = [np.zeros((num_samples, ) + data.shape[1:])]

    rseed = np.random.randint(0, 1e6)
    np.random.seed(rseed)
    phis = [np.zeros((1,) + data.shape[1:])]
    # phi_vars = [np.zeros((output_batch,) + X.shape)]
    for j in range(1):
        for k in range(num_samples):
            rind = np.random.choice(data_iterator().size)
            t = np.random.uniform()
            im = data_iterator()._data_source._get_data(rind)[0]
            x = X.copy()
            samples_input[0][k] = (t * x + (1 - t) * im.copy()).copy()
            if interim_layer == 0:
                samples_delta[0][k] = (x - im.copy()).copy()
            else:
                samples_delta[0][k] = get_interim_input(
                    model, samples_input[0][k], interim_layer)[0]

        grads = []

        for b in range(0, num_samples, batch_size):
            batch_last = min(b + batch_size, num_samples)
            batch = samples_input[0][b:batch_last].copy()
            grads.append(gradient(model, label, batch,
                                  batch_last - b, interim_layer))
        grad = [np.concatenate([g[0] for g in grads], 0)]
        samples = grad[0] * samples_delta[0]
        phis[0][j] = samples.mean(0)

    output_phis.append(phis[0])

    img = X.copy()
    height = img.shape[1]
    width = img.shape[2]
    ratio = 5 / height

    fig_size = np.array([width * ratio, 5])
    fig, ax = plt.subplots(figsize=fig_size, dpi=1 / ratio)
    shap_plot = output_phis[0][0].sum(0)

    if img.max() > 1:
        img = img / 255.
    if img.shape[0] == 3:
        img_gray = (0.2989 * img[0, :, :] + 0.5870 *
                    img[1, :, :] + 0.1140 * img[2, :, :])
    else:
        img_gray = img.reshape(img.shape[1:])

    abs_phis = np.abs(output_phis[0].sum(1)).flatten()
    max_border = np.nanpercentile(abs_phis, 99.9)
    min_border = -np.nanpercentile(abs_phis, 99.9)

    ax.imshow(img_gray, cmap=plt.get_cmap('gray'), alpha=0.15,
              extent=(-1, shap_plot.shape[1], shap_plot.shape[0], -1))
    im = ax.imshow(shap_plot, cmap=red_blue_map(),
                   vmin=min_border, vmax=max_border)
    ax.axis("off")

    fig.savefig(output)
    fig.clf()
    plt.close()
