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

import csv
import os
import h5py
import numpy as np
from utils.file import read_csv, add_info_to_csv, save_to_csv


def get_labels(csv_file):
    with open(csv_file) as f:
        reader = csv.reader(f)
        l = [row for row in reader]

    header_contents = l[0]

    for content in header_contents:
        if "y:label" in content:
            label_list = content.split(";")[1:]
        else:
            label_list = []

    return label_list


def sigmoid_func(x):
    return 1.0 / (1.0 + np.exp(-x))


def visuallize_infl(args, info):

    with h5py.File(info, "r") as hf:
        train_feature = hf["feature"]["train"][:]
        train_output = hf["output"]["train"][:]
        train_label = hf["label"]["train"][:]
        train_input_path = hf["input"]["train"][:]

        val_feature = hf["feature"]["test"][:]
        val_output = hf["output"]["test"][:]
        val_label = hf["label"]["test"][:]
        val_input_path = hf["input"]["test"][:]

    weight_matrix = np.load(os.path.join(
        args.monitor_path, "weight_matrix.npy"))

    if val_output.shape[1] == 1:
        sigmoid_ = sigmoid_func(val_output)
        output_test_labels = np.where(sigmoid_ > 0.5, 1, 0)
    else:
        output_test_labels = np.argmax(val_output, axis=1)
    label_name_list = get_labels(args.input_val)
    test_point_random = np.random.randint(0, len(val_label), args.num_samples)
    top_k = args.top_k

    header = ["test_sample", "label; pred"]
    header.extend(["positive_" + str(i+1) for i in range(top_k)])
    header.extend(["negative_" + str(i+1) for i in range(top_k)])
    info_list = [header]

    for test_ind in test_point_random:
        info = []
        target_class = output_test_labels[test_ind]
        tmp = weight_matrix[:, target_class] * np.sum(
            train_feature * val_feature[test_ind, :], axis=1
        )

        pos_idx = np.flip(np.argsort(tmp), axis=0)
        neg_idx = np.argsort(tmp)

        try:
            info.extend([val_input_path[test_ind], label_name_list[val_label[test_ind]]
                        + "; " + label_name_list[output_test_labels[test_ind]] + "(" + str(val_label[test_ind]) + "; " + str(output_test_labels[test_ind]) + ")"])
        except IndexError:
            info.extend([val_input_path[test_ind], str(val_label[test_ind]) + "; " +
                        str(output_test_labels[test_ind])])

        info.extend([train_input_path[id]
                    for k, id in enumerate(pos_idx) if k < top_k])
        info.extend([train_input_path[id]
                    for k, id in enumerate(neg_idx) if k < top_k])

        info_list.append(info)

    with open(args.output, "w") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(info_list)
