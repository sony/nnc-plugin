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

import argparse
import functools
from json import load
import os
from tqdm import tqdm
from distutils.util import strtobool

from utils.model import get_context

import h5py
import nnabla as nn
import nnabla.functions as F
import numpy as np
from nnabla.ext_utils import get_extension_context
from nnabla.utils.data_iterator import data_iterator
from nnabla.utils.nnp_graph import NnpLoader
import nnabla.utils.load as load

from .datasource import get_datasource

from collections import OrderedDict


class get_middle_variables:
    def __init__(self):
        self.middle_vars_dict = OrderedDict()
        self.middle_layer_count_dict = OrderedDict()

    def __call__(self, f):
        if f.name in self.middle_layer_count_dict:
            self.middle_layer_count_dict[f.name] += 1
        else:
            self.middle_layer_count_dict[f.name] = 1
        key = f.name + "_{}".format(self.middle_layer_count_dict[f.name])
        self.middle_vars_dict[key] = f.outputs[0]


def ensure_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def load_data(input_path, normalize):
    print("Now Loading Datasets")
    data_source = get_datasource(filename=input_path, normalize=normalize)
    iterator = data_iterator(data_source, batch_size, None, False, False)
    return data_source, iterator


def feature_extractor(dataloaders_dict, image_valid, pred_hidden, pred):
    phase_feature, phase_preds, phase_input = {}, {}, {}

    for phase, dataloader in dataloaders_dict.items():
        img_path_list = []
        extracted_feature, preds = [], []
        iteration = int(dataloader._size / batch_size)
        correct, num_samples = 0, 0

        for j in tqdm(range(iteration)):
            image, label, *_extra_info = dataloader.next()
            _, ds_idxes = _extra_info
            image_valid.d = image
            num_samples += len(image)
            pred.forward(clear_buffer=True)

            pred_ = np.argmax(pred.d, axis=1)
            pred_hidden.forward(clear_buffer=True)

            extracted_feature.append(pred_hidden.d.reshape(len(image), -1))
            preds.append(pred.d)
            ds_idx = ds_idxes.astype(int)

            abs_img_path = [
                dataloader._data_source.get_abs_filepath_to_data(i[0]) for i in ds_idx
            ]
            img_path_list.extend(abs_img_path)

        concat = np.concatenate(extracted_feature, 0)
        preds = np.concatenate(preds, 0)

        phase_feature[phase] = concat
        phase_preds[phase] = preds
        phase_input[phase] = img_path_list

    return phase_feature, phase_preds, phase_input


def generate_feature(args):
    ctx = get_extension_context(
        ext_name="cudnn", device_id=args.device_id, type_config=args.type_config
    )
    nn.set_default_context(ctx)

    global batch_size
    batch_size = args.batch_size
    train_source, train_loader = load_data(args.input_train, args.normalize)
    val_source, val_loader = load_data(args.input_val, args.normalize)
    phase_source = {"train": train_source, "test": val_source}

    print("Now Loading Model")
    nnp = NnpLoader(args.model)

    class ForwardConfig:
        pass

    config = ForwardConfig
    info = load.load([args.model],
                     prepare_data_iterator=False, batch_size=1)
    config.executors = info.executors.values()
    executor = list(config.executors)[0]

    model = nnp.get_network(executor.network.name, batch_size)

    image_valid = model.inputs[list(model.inputs.keys())[0]]
    label_val = nn.Variable((batch_size, 1))
    softmax = model.outputs[list(model.outputs.keys())[0]]

    GET_MIDDLE_VARIABLES_CLASS = get_middle_variables()
    softmax.visit(GET_MIDDLE_VARIABLES_CLASS)
    middle_vars = GET_MIDDLE_VARIABLES_CLASS.middle_vars_dict
    model_variables = [v for v in middle_vars.values()]
    model_variables_name = [v for v in middle_vars.keys()]

    for ind, name in enumerate(model_variables_name):
        if "Affine" in name:
            affine_output_ind = ind
            affine_input_ind = ind - 1

    pred_hidden = model_variables[affine_input_ind]
    pred = model_variables[affine_output_ind]

    affine_weight = []
    last_layer = [name for name in nn.get_parameters().keys()
                  ][-1].split("/")[0]

    for name, param in nn.get_parameters().items():
        if last_layer in name:
            affine_weight.append(param.d)

    dataloaders_dict = {"train": train_loader, "test": val_loader}

    phase_feature, phase_output, phase_input = feature_extractor(
        dataloaders_dict, image_valid, pred_hidden, pred
    )

    with h5py.File(os.path.join(args.monitor_path, "info.h5"), "w") as hf:

        hf.create_group("label")
        for name, data_source_phase in phase_source.items():
            hf["label"].create_dataset(
                name, data=data_source_phase._get_labels().astype(int)
            )

        hf.create_group("param")
        for name, param in zip(["weight", "bias"], affine_weight):
            hf["param"].create_dataset(name, data=param)

        hf.create_group("feature")
        for phase, feature in phase_feature.items():
            hf["feature"].create_dataset(phase, data=feature)

        hf.create_group("output")
        for phase, output in phase_output.items():
            hf["output"].create_dataset(phase, data=output)

        hf.create_group("input")
        for phase, input in phase_input.items():
            input = np.array(input, dtype=h5py.special_dtype(vlen=str))
            hf["input"].create_dataset(phase, data=input)
