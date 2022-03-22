# Copyright 2022 Sony Group Corporation.
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
import math
import numpy as np
from tqdm import tqdm
import skimage.segmentation
from nnabla import logger
import nnabla.utils.load as load
from nnabla.utils.image_utils import imread, imsave
from nnabla.utils.cli.utility import let_data_to_variable
from nnabla.utils.data_iterator import data_iterator_csv_dataset
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S


def ridge(dataset):
    import nnabla_ext.cpu
    ctx = nnabla_ext.cpu.context()
    with nn.context_scope(ctx):
        dataset = np.array(dataset)
        nn.clear_parameters()
        x = nn.Variable((int(math.sqrt(dataset.shape[0])), dataset[0][0].size))
        t = nn.Variable((x.shape[0], 1))
        y = PF.affine(x, 1, name='affine')
        loss = F.squared_error(y, t)
        mean_loss = F.mean(loss)

        solver = S.Momentum()
        solver.set_parameters(nn.get_parameters())
        for iter in range(100 * int(math.sqrt(dataset.shape[0]))):  # 100 epoch
            np.random.shuffle(dataset)
            x.d = np.stack(dataset[:x.shape[0], 0]).reshape(x.shape)
            t.d = np.stack(dataset[:x.shape[0], 1]).reshape(t.shape)
            solver.zero_grad()
            mean_loss.forward()
            mean_loss.backward()
            solver.weight_decay(0.01)
            solver.update()

    return nn.get_parameters()['affine/affine/W'].d.flatten()


def lime_func(args):
    class ForwardConfig:
        pass
    # Load model
    info = load.load([args.model], prepare_data_iterator=False, batch_size=1)

    config = ForwardConfig
    config.global_config = info.global_config

    config.executors = info.executors.values()

    config.networks = []
    if len(config.executors) < 1:
        logger.critical('Executor is not found in {}.'.format(args.model))
        return
    executor = list(config.executors)[0]
    if len(config.executors) > 1:
        logger.log(
            99, 'Only the first executor {} is used in the LIME calculation.'.format(executor.name))

    if executor.network.name in info.networks.keys():
        config.networks.append(info.networks[executor.network.name])
    else:
        logger.critical('Network {} is not found in {}.'.format(
            executor.network.name, args.model))
        return

    # Prepare variable
    input_variable, data_name = list(executor.dataset_assign.items())[0]
    output_variable = list(executor.output_assign.keys())[0]

    # Load image
    im = imread(args.image)
    slic = skimage.segmentation.slic(im, n_segments=args.num_segments)
    if not executor.no_image_normalization:
        im = im / 255.0
    if len(im.shape) == 3:
        im = im.transpose(2, 0, 1)
    else:
        im = im.reshape((1,) + im.shape)

    mask_and_result = []
    # Sampling
    logger.log(99, 'Sampling ...')
    np.random.seed(0)
    with tqdm(total=args.num_samples) as pbar:
        while len(mask_and_result) < args.num_samples:
            x = input_variable.variable_instance
            mask = np.random.uniform(
                size=(x.shape[0], args.num_segments)) >= 0.5
            im_mask = np.array([mask1[slic]
                                for mask1 in mask]).astype(np.uint8)
            data = im_mask.reshape(
                x.shape[0], 1, x.shape[2], x.shape[3]) * im.reshape(1, x.shape[1], x.shape[2], x.shape[3])

            # input data
            let_data_to_variable(input_variable.variable_instance,
                                 data,
                                 data_name=data_name, variable_name=input_variable.name)

            # Generate data
            for v, generator in executor.generator_assign.items():
                v.variable_instance.d = generator(v.shape)

            # Forward
            executor.forward_target.forward(clear_buffer=True)

            for m, probability in zip(mask, output_variable.variable_instance.d):
                mask_and_result.append([m, probability[args.class_index]])

            pbar.update(x.shape[0])

    logger.log(99, 'Evaluating ...')
    weight = ridge(mask_and_result)
    max_indices = np.argpartition(-weight,
                                  args.num_segments_2)[:args.num_segments_2]
    result = np.zeros(weight.shape)
    result[max_indices] = 1

    # Generate output image
    result_mask = result[slic].astype(
        np.uint8).reshape(1, im.shape[1], im.shape[2])
    if not executor.no_image_normalization:
        im = im * 255.0
    im = im * result_mask + \
        np.ones(im.shape, np.uint8) * 192 * np.logical_not(result_mask)
    result = im.transpose(1, 2, 0).astype(np.uint8)
    if result.shape[-1] == 1 and len(result.shape) == 3:
        result = result.reshape(result.shape[0:2])
    return result


def lime_batch_func(args):
    class ForwardConfig:
        pass
    # Load model
    info = load.load([args.model], prepare_data_iterator=False, batch_size=1)

    config = ForwardConfig
    config.global_config = info.global_config

    config.executors = info.executors.values()

    config.networks = []
    if len(config.executors) < 1:
        logger.critical('Executor is not found in {}.'.format(args.model))
        return
    executor = list(config.executors)[0]
    if len(config.executors) > 1:
        logger.log(99, 'Only the first executor {} is used in the LIME calculation.'.format(
            executor.name))

    if executor.network.name in info.networks.keys():
        config.networks.append(info.networks[executor.network.name])
    else:
        logger.critical('Network {} is not found in {}.'.format(
            executor.network.name, args.model))
        return

    # Prepare variable
    input_variable, data_name = list(executor.dataset_assign.items())[0]
    output_variable = list(executor.output_assign.keys())[0]

    # check
    csv_file = np.loadtxt(args.input, delimiter=",", dtype=str)
    header = [item.split(":")[0] for item in csv_file[0]]
    classes = csv_file[1:, header.index(args.label_variable)]
    num_classes = np.unique(classes).size

    output_size = output_variable.variable_instance.d.shape[1]
    is_binary_classification = num_classes != output_size

    # Load dataset
    data_iterator = (lambda: data_iterator_csv_dataset(
        uri=args.input,
        batch_size=1,
        shuffle=False,
        normalize=not executor.no_image_normalization,
        with_memory_cache=False,
        with_file_cache=False))

    # Prepare output
    data_output_dir = os.path.splitext(args.output)[0]

    # Data loop
    with data_iterator() as di:
        index = 0
        file_names = []
        while index < di.size:
            # Load data
            data = di.next()
            im = data[di.variables.index(args.input_variable)]
            im = im[0].transpose(1, 2, 0)
            slic = skimage.segmentation.slic(im, n_segments=args.num_segments)
            im = im.transpose(2, 0, 1)
            if is_binary_classification:
                label = 0
            else:
                label = data[di.variables.index(args.label_variable)]
                label = label.reshape((label.size,))
                if label.size > 1:
                    label = np.argmax(label)
                else:
                    label = label[0]

            mask_and_result = []
            # Sampling
            np.random.seed(0)
            while len(mask_and_result) < args.num_samples:
                x = input_variable.variable_instance
                mask = np.random.uniform(
                    size=(x.shape[0], args.num_segments)) >= 0.5
                im_mask = np.array([mask1[slic]
                                    for mask1 in mask]).astype(np.uint8)
                data = im_mask.reshape(
                    x.shape[0], 1, x.shape[2], x.shape[3]) * im.reshape(1, x.shape[1], x.shape[2], x.shape[3])

                # input data
                let_data_to_variable(input_variable.variable_instance,
                                     data,
                                     data_name=data_name, variable_name=input_variable.name)

                # Generate data
                for v, generator in executor.generator_assign.items():
                    v.variable_instance.d = generator(v.shape)

                # Forward
                executor.forward_target.forward(clear_buffer=True)

                for m, probability in zip(mask, output_variable.variable_instance.d):
                    try:
                        mask_and_result.append([m, probability[int(label)]])
                    except IndexError as e:
                        if index != 0:
                            pbar.close()
                        logger.critical(e)
                        return

            weight = ridge(mask_and_result)
            max_indices = np.argpartition(-weight,
                                          args.num_segments_2)[:args.num_segments_2]
            result = np.zeros(weight.shape)
            result[max_indices] = 1

            # Generate output image
            result_mask = result[slic].astype(
                np.uint8).reshape(1, im.shape[1], im.shape[2])
            if not executor.no_image_normalization:
                im = im * 255.0
            im = im * result_mask + \
                np.ones(im.shape, np.uint8) * 192 * np.logical_not(result_mask)
            result = im.transpose(1, 2, 0).astype(np.uint8)
            if result.shape[-1] == 1 and len(result.shape) == 3:
                result = result.reshape(result.shape[0:2])

            # Output result image
            file_name = os.path.join(data_output_dir, '{:04d}'.format(
                index // 1000), '{}.png'.format(index))
            directory = os.path.dirname(file_name)
            try:
                os.makedirs(directory)
            except OSError:
                pass  # python2 does not support exists_ok arg
            imsave(file_name, result)

            file_names.append(file_name)
            if index == 0:
                pbar = tqdm(total=di.size)
            index += 1
            pbar.update(1)

        pbar.close()

    return file_names
