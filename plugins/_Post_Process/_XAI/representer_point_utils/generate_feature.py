# Copyright (c) 2022 Sony Group Corporation. All Rights Reserved.
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
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> c5bf7bc (fix to pretrained nnc)
from json import load
import os
from tqdm import tqdm
from distutils.util import strtobool

from utils.model import get_context

import h5py
import nnabla as nn
<<<<<<< HEAD
import nnabla.communicators as C
<<<<<<< HEAD
=======
import os
from distutils.util import strtobool

import h5py
import nnabla as nn
>>>>>>> fe1dfc3 (first commit)
=======
>>>>>>> a1273c6 (first commit)
=======
>>>>>>> b80d247 (bug fix)
import nnabla.functions as F
import numpy as np
from nnabla.ext_utils import get_extension_context
from nnabla.utils.data_iterator import data_iterator
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> c5bf7bc (fix to pretrained nnc)
from nnabla.utils.nnp_graph import NnpLoader
import nnabla.utils.load as load

from .datasource import get_datasource

<<<<<<< HEAD
<<<<<<< HEAD
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
        key = f.name + '_{}'.format(self.middle_layer_count_dict[f.name])
        self.middle_vars_dict[key] = f.outputs[0]
=======
=======
bs_valid = 16
>>>>>>> c5bf7bc (fix to pretrained nnc)

=======
>>>>>>> a1273c6 (first commit)
from collections import OrderedDict


<<<<<<< HEAD
bs_valid = 100
CHECKPOINTS_PATH_FORMAT = "params_270.h5"
>>>>>>> fe1dfc3 (first commit)
=======
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
>>>>>>> c5bf7bc (fix to pretrained nnc)


def ensure_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


<<<<<<< HEAD
<<<<<<< HEAD
def load_data(input_path, normalize):
    print("Now Loading Datasets")
    data_source = get_datasource(filename=input_path, normalize=normalize)
    iterator = data_iterator(data_source, batch_size, None, False, False)
    return data_source, iterator


=======
>>>>>>> fe1dfc3 (first commit)
=======
def load_data(input_path, normalize):
    print("Now Loading Datasets")
    data_source = get_datasource(filename=input_path, normalize=normalize)
    iterator = data_iterator(data_source, batch_size, None, False, False)
    return data_source, iterator


>>>>>>> c5bf7bc (fix to pretrained nnc)
def feature_extractor(dataloaders_dict, image_valid, pred_hidden, pred):
    phase_feature, phase_preds, phase_input = {}, {}, {}

    for phase, dataloader in dataloaders_dict.items():
        img_path_list = []
        extracted_feature, preds = [], []
<<<<<<< HEAD
<<<<<<< HEAD
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

<<<<<<< HEAD
<<<<<<< HEAD
        print('Overall Accuracy: ', correct / num_samples)
=======
        iteration = int(dataloader._size / bs_valid)
        for j in tqdm(range(iteration)):
=======
        iteration = int(dataloader._size / batch_size)
        correct, num_samples = 0, 0
>>>>>>> a1273c6 (first commit)

        for j in tqdm(range(iteration)):
            if j > 10:
                continue
            image, label, *_extra_info = dataloader.next()
            image_valid.d = image
            num_samples += len(image)
            pred.forward(clear_buffer=True)
            pred_ = np.argmax(pred.d, axis=1)
            correct += len(np.where(pred_ == label.reshape(-1))[0])

            pred_hidden.forward(clear_buffer=True)
            extracted_feature.append(pred_hidden.d.reshape(len(image), -1))
            preds.append(pred.d)
<<<<<<< HEAD
>>>>>>> fe1dfc3 (first commit)
=======
        print('Overall Accuracy: ', correct / num_samples)
>>>>>>> a1273c6 (first commit)
=======
        print("Overall Accuracy: ", correct / num_samples)
>>>>>>> b80d247 (bug fix)
=======
>>>>>>> 381b077 (fix network architecture bug)
        concat = np.concatenate(extracted_feature, 0)
        preds = np.concatenate(preds, 0)

        phase_feature[phase] = concat
        phase_preds[phase] = preds
        phase_input[phase] = img_path_list

    return phase_feature, phase_preds, phase_input


<<<<<<< HEAD
<<<<<<< HEAD
def generate_feature(args):
    ctx = get_extension_context(ext_name="cudnn",
                                device_id=args.device_id,
                                type_config=args.type_config)

    comm = C.MultiProcessDataParalellCommunicator(ctx)
    comm.init()
    n_devices = comm.size
    mpi_rank = comm.rank
    device_id = mpi_rank
    ctx = get_extension_context('cudnn', device_id=device_id)

    nn.set_default_context(ctx)

    global batch_size
    batch_size = args.batch_size
    train_source, train_loader = load_data(args.input_train, args.normalize)
    _, val_loader = load_data(args.input_val, args.normalize)

    print("Now Loading Model")
    nnp = NnpLoader(args.model_path)

    class ForwardConfig:
        pass

    config = ForwardConfig
    info = load.load([args.model_path],
                     prepare_data_iterator=False,
                     batch_size=1)
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

    pred_hidden = model_variables[-3]
    pred = model_variables[-1]

    affine_weight = []
    last_layer = [name
                  for name in nn.get_parameters().keys()][-1].split("/")[0]

=======
def main():
    extension_module = args.context
    ctx = get_extension_context(
        extension_module, device_id=args.device_id, type_config=args.type_config
    )
=======
def generate_feature(args):
<<<<<<< HEAD
<<<<<<< HEAD
    ctx = get_context(device_id=args.device_id)
>>>>>>> c5bf7bc (fix to pretrained nnc)
=======
    ctx = get_extension_context(ext_name="cudnn",
                                device_id=args.device_id,
                                type_config=args.type_config)

    comm = C.MultiProcessDataParalellCommunicator(ctx)
    comm.init()
    n_devices = comm.size
    mpi_rank = comm.rank
    device_id = mpi_rank
    ctx = get_extension_context('cudnn', device_id=device_id)

>>>>>>> a1273c6 (first commit)
=======
    ctx = get_extension_context(
        ext_name="cudnn", device_id=args.device_id, type_config=args.type_config
    )
>>>>>>> b80d247 (bug fix)
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
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    last_layer = [name for name in nn.get_parameters().keys()
                  ][-1].split("/")[0]
>>>>>>> fe1dfc3 (first commit)
=======
    last_layer = [name
                  for name in nn.get_parameters().keys()][-1].split("/")[0]
=======
    last_layer = [name for name in nn.get_parameters().keys()][-1].split("/")[0]
>>>>>>> b80d247 (bug fix)
=======
    last_layer = [name for name in nn.get_parameters().keys()
                  ][-1].split("/")[0]
>>>>>>> 6da9a2f (fix copyright)

>>>>>>> c5bf7bc (fix to pretrained nnc)
    for name, param in nn.get_parameters().items():
        if last_layer in name:
            affine_weight.append(param.d)

<<<<<<< HEAD
<<<<<<< HEAD
    dataloaders_dict = {"train": train_loader, "test": val_loader}

    phase_feature, phase_output = feature_extractor(dataloaders_dict,
                                                    image_valid, pred_hidden,
                                                    pred)

    with h5py.File(os.path.join(args.monitor_path, "info.h5"), "w") as hf:
        hf.create_dataset("label", data=train_source._get_labels().astype(int))
=======
    data_save_dir = (
        "./data/info/shuffle" if args.shuffle_label else "./data/info/no_shuffle"
    )
    ensure_dir(data_save_dir)

    data_source_train = Cifar10NumpySource(X_train, Y_train)
    train_loader = data_iterator(
        data_source_train, bs_valid, None, False, False)
    data_source_val = Cifar10NumpySource(X_val, Y_val)
    val_loader = data_iterator(data_source_val, bs_valid, None, False, False)
    dataloaders_dict = {"train": train_loader, "test": val_loader}

    phase_feature, phase_output = feature_extractor(
        dataloaders_dict, image_valid, pred_hidden, pred
    )

    with h5py.File(os.path.join(data_save_dir, "info.h5"), "w") as hf:
        hf.create_dataset("label", data=Y_train)
>>>>>>> fe1dfc3 (first commit)
        hf.create_group("param")
        for name, param in zip(["weight", "bias"], affine_weight):
            hf["param"].create_dataset(name, data=param)
        hf.create_group("feature")
        for phase, feature in phase_feature.items():
            print(feature.shape)
            hf["feature"].create_dataset(phase, data=feature)
        hf.create_group("output")
        for phase, output in phase_output.items():
            hf["output"].create_dataset(phase, data=output)

<<<<<<< HEAD
=======
    dataloaders_dict = {"train": train_loader, "test": val_loader}

    phase_feature, phase_output, phase_input = feature_extractor(
        dataloaders_dict, image_valid, pred_hidden, pred
    )

<<<<<<< HEAD
>>>>>>> c5bf7bc (fix to pretrained nnc)
=======
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

<<<<<<< HEAD
>>>>>>> 0c7f1e3 (save contents to h5 file)
    return {
        'label': train_source._get_labels(),
        'param': affine_weight,
        'feature': phase_feature,
        'output': phase_output
    }
<<<<<<< HEAD
=======
    print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature generator")
    parser.add_argument("--shuffle_label", type=strtobool)
    parser.add_argument("--device_id", "-d", type=str, default="0")
    parser.add_argument("--type_config", "-t", type=str, default="float")
    parser.add_argument("--context", "-c", type=str, default="cudnn")
    args = parser.parse_args()
    main()
>>>>>>> fe1dfc3 (first commit)
=======
>>>>>>> c5bf7bc (fix to pretrained nnc)
=======
        hf.create_group("input")
        for phase, input in phase_input.items():
            input = np.array(input, dtype=h5py.special_dtype(vlen=str))
            hf["input"].create_dataset(phase, data=input)
>>>>>>> 7d3a19a (added visualizing func)
