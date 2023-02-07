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
import numpy as np
import tqdm
import nnabla.utils.load as load
from nnabla.utils.data_iterator import data_iterator_csv_dataset
from nnabla.utils.image_utils import imread
from .utils import get_executor, red_blue_map, gradient, \
    get_interim_input, plot_shap


def shap_func(args):
    executor = get_executor(args.model)

    # Load model
    info = load.load([args.model], prepare_data_iterator=False,
                     batch_size=args.batch_size)

    # Data source
    data_iterator = (lambda: data_iterator_csv_dataset(
        uri=args.input,
        batch_size=1,
        shuffle=False,
        normalize=not executor.no_image_normalization,
        with_memory_cache=False,
        with_file_cache=False))

    X = imread(args.image)
    if not executor.no_image_normalization:
        X = X / 255.0
    if len(X.shape) == 3:
        X = X.transpose(2, 0, 1)
    else:
        X = X.reshape((1,) + X.shape)

    with data_iterator() as di:
        plot_shap(model=args.model, info=info, X=X,
                  label=args.class_index, output=args.output,
                  interim_layer=args.interim_layer,
                  num_samples=args.num_samples,
                  data_iterator=di, batch_size=args.batch_size,
                  red_blue_map=red_blue_map, gradient=gradient,
                  get_interim_input=get_interim_input)


def shap_batch_func(args):

    executor = get_executor(args.model)

    # Load model
    info = load.load([args.model], prepare_data_iterator=False,
                     batch_size=args.batch_size)

    # Prepare variable
    output_variable = list(executor.output_assign.keys())[0]

    # Data source
    data_iterator = (lambda: data_iterator_csv_dataset(
        uri=args.input,
        batch_size=1,
        shuffle=False,
        normalize=not executor.no_image_normalization,
        with_memory_cache=False,
        with_file_cache=False))

    # Prepare output
    data_output_dir = os.path.splitext(args.output)[0]

    # check
    csv_file = np.loadtxt(args.input, delimiter=",", dtype=str)
    header = [item.split(":")[0] for item in csv_file[0]]
    classes = csv_file[1:, header.index(args.label_variable)]
    num_classes = np.unique(classes).size

    output_size = output_variable.variable_instance.d.shape[1]
    is_binary_classification = num_classes != output_size

    # Data loop
    with data_iterator() as di:
        index = 0
        file_names = []
        while index < di.size:
            file_name = os.path.join(data_output_dir, '{:04d}'.format(
                                     index // 1000), '{}.png'.format(index))
            directory = os.path.dirname(file_name)
            try:
                os.makedirs(directory)
            except OSError:
                pass  # python2 does not support exists_ok arg
            # Load data
            data = di.next()
            im = data[di.variables.index(args.input_variable)]
            im = im.reshape((im.shape[1], im.shape[2], im.shape[3]))
            if is_binary_classification:
                label = 0
            else:
                label = data[di.variables.index(args.label_variable)]
                label = label.reshape((label.size,))
                if label.size > 1:
                    label = np.argmax(label)
                else:
                    label = label[0]
            if index == 0:
                pbar = tqdm.tqdm(total=di.size)

            plot_shap(model=args.model, info=info, X=im, label=label,
                      output=file_name, interim_layer=args.interim_layer,
                      num_samples=args.num_samples, data_iterator=di,
                      batch_size=args.batch_size,
                      red_blue_map=red_blue_map, gradient=gradient,
                      get_interim_input=get_interim_input)

            file_names.append(file_name)
            index += 1
            pbar.update(1)

    pbar.close()

    return file_names
