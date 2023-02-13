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

import argparse
import os
import random
from distutils.util import strtobool


import csv
import h5py
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import numpy as np
from nnabla.ext_utils import get_extension_context
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt


def label_shuffle(labels_, shuffle_rate, seed):
    random.seed(seed)
    num_cls = int(np.max(labels_)) + 1
    raw_label = labels_.copy()
    extract_num = int(len(labels_) * shuffle_rate // 10)
    for i in range(num_cls):
        extract_ind = np.where(raw_label == i)[0]
        labels = [j for j in range(num_cls)]
        labels.remove(i)  # candidate of shuffle label
        artificial_label = [
            labels[int(i) % (num_cls - 1)] for i in range(int(extract_num))
        ]
        artificial_label = np.array(
            random.sample(artificial_label, len(artificial_label))
        ).astype("float32")
        convert_label = np.array([i for _ in range(len(extract_ind))])
        convert_label[-extract_num:] = artificial_label

        labels_[extract_ind] = convert_label.reshape(-1, 1)

    return raw_label, labels_


def loss_function(args, pred, y_output, N):
    # calculate cross entropy loss

    log_softmax = F.log_softmax(pred, axis=1)
    Phi = F.sum(-y_output * log_softmax)
    # calculate l2 norm of affine layer
    l2 = 0
    for param in nn.get_parameters().values():
        l2 += F.sum(param**2)
    loss = l2 * args.lmbd + Phi / N

    return loss, Phi, l2, log_softmax


def backtracking_line_search(grad_norm, x, y, loss, N, val, l2, threshold=1e-10):
    t = 10.0
    beta = 0.5
    params = nn.get_parameters().values()
    p_data_org_list = [F.identity(p.data) for p in params]
    p_grad_org_list = [F.identity(p.grad) for p in params]

    while True:
        for p, p_data_org, p_grad_org in zip(params, p_data_org_list, p_grad_org_list):
            p.data.copy_from(p_data_org - t * p_grad_org)

        loss.forward()
        if t < threshold:
            print("t too small")
            break
        if (loss.d - val + t * grad_norm.data**2 / 2) >= 0:
            t = beta * t
        else:
            break

    return params


def calculate_alpha(
    args,
    parameter_list,
    X,
    Y,
    Y_label,
    feature_valid,
    solver,
    output_valid,
    softmax_valid,
    pred,
    loss,
    phi,
    l2,
    lg,
):
    print("## Now Score Calculating")
    min_loss = 10000.0
    feature_valid.d = X
    output_valid.d = Y

    for epoch in range(args.epoch):
        phi_loss = 0

        loss.forward()
        solver.zero_grad()
        loss.backward()

        phi_loss = phi.d / len(X)
        temp_W = parameter_list

        grad_loss = F.add_n(
            *[F.mean(F.abs(p.grad)) for p in nn.get_parameters().values()]
        )
        grad_norm = F.add_n(*[F.norm(p.grad)
                            for p in nn.get_parameters().values()])

        if grad_loss.data < min_loss:
            if epoch == 0:
                init_grad = grad_loss.data
            min_loss = grad_loss.data
            best_W = temp_W
            if min_loss < init_grad / 200:
                print("stopping criteria reached in epoch :{}".format(epoch))
                break
        parameter_list = backtracking_line_search(
            grad_norm, X, softmax_valid.d, loss, len(X), loss.d, l2
        )
        if epoch % 100 == 0:
            print(
                "Epoch:{:4d}\tloss:{}\tphi_loss:{}\tl2(lmbd):{}\tgrad:{}".format(
                    epoch, loss.d, phi_loss, args.lmbd * l2.d, grad_loss.data
                )
            )

    for weight, param in zip(nn.get_parameters().values(), best_W):
        weight.data.copy_from(param.data)

    softmax_value = F.softmax(pred)
    softmax_value.forward()
    # derivative of softmax cross entropy
    weight_matrix = softmax_value.d - softmax_valid.d
    weight_matrix = np.divide(weight_matrix, (-2.0 * args.lmbd * len(Y)))
    np.save(os.path.join(args.monitor_path, "weight_matrix.npy"), weight_matrix)

    # computer alpha
    alpha = []
    print(weight_matrix.shape)
    for ind, label in enumerate(Y_label.reshape(-1)):
        alpha.append(float(weight_matrix[ind, int(label)]))
    alpha = np.abs(np.array(alpha))
    np.save(os.path.join(args.monitor_path, "alpha_vgg_nnabla_score.npy"), alpha)

    # calculate correlation
    w = np.matmul(X.T, weight_matrix)
    temp = np.matmul(X, w)
    softmax_value = F.softmax(nn.Variable.from_numpy_array(temp))
    softmax_value.forward()
    y_p = softmax_value.d

    print(
        "L1 difference between ground truth prediction and prediction by representer theorem decomposition"
    )
    print(np.mean(np.abs(softmax_valid.d - y_p)))

    print(
        "pearson correlation between ground truth  prediction and prediciton by representer theorem"
    )
    corr, _ = pearsonr(Y.reshape(-1), y_p.reshape(-1))
    print(corr)

    return weight_matrix


def compute_score(args, info):

    ctx = get_extension_context(
        ext_name="cudnn", device_id=args.device_id, type_config=args.type_config
    )

    nn.set_default_context(ctx)

    with h5py.File(info, "r") as hf:
        train_feature = hf["feature"]["train"][:]
        test_feature = hf["feature"]["test"][:]
        train_output = hf["output"]["train"][:]
        test_output = hf["output"]["test"][:]
        Y_label = hf["label"]["train"][:][: len(train_feature)]
        parameter_list = [hf["param"]["weight"][:], hf["param"]["bias"][:]]

    feature_dict = {"train": train_feature, "test": test_feature}
    output_dict = {"train": train_output, "test": test_output}
    output_dict_softmax = dict()
    for phase, pre_act in output_dict.items():
        softmax_ = F.softmax(nn.Variable.from_numpy_array(pre_act))
        softmax_.forward()
        output_dict_softmax[phase] = softmax_.d
    feature_shape = train_feature.shape
    n_cls = train_output.shape[-1]

    nn.clear_parameters()
    # computation graph
    feature_valid = nn.Variable((feature_shape[0], feature_shape[1]))
    output_valid = nn.Variable((feature_shape[0], n_cls))
    softmax_valid = F.softmax(output_valid)

    with nn.parameter_scope("final_layer"):
        pred = PF.affine(feature_valid, n_cls)

    loss, phi, l2, lg = loss_function(
        args, pred, softmax_valid, len(train_feature))

    # parameter initialized
    for weight, param in zip(nn.get_parameters().values(), parameter_list):
        weight.data.copy_from(nn.NdArray.from_numpy_array(param))

    solver = S.Sgd(lr=1.0)
    solver.set_parameters(nn.get_parameters())
    for ind, (name, param) in enumerate(nn.get_parameters().items()):
        param.grad.zero()
        param.need_grad = True

    weight_matrix = calculate_alpha(
        args,
        parameter_list,
        train_feature,
        train_output,
        Y_label,
        feature_valid,
        solver,
        output_valid,
        softmax_valid,
        pred,
        loss,
        phi,
        l2,
        lg,
    )

    predicted_dict = {}

    for phase, feature in feature_dict.items():
        w = np.matmul(train_feature.T, weight_matrix)
        temp = np.matmul(feature, w)
        softmax = F.softmax(nn.Variable.from_numpy_array(temp))
        softmax.forward()
        predicted_dict[phase] = softmax.d

    np.random.seed(401)
    train_rand_idx = np.random.choice(50000, 2000, replace=False)
    np.random.seed(4)
    test_rand_idx = np.random.choice(10000, 1000, replace=False)
    rand_idx_dict = {"train": train_rand_idx, "test": test_rand_idx}

    for ind, phase in enumerate(["train", "test"]):

        info_list = [["actual_output", "approximated_output"]]
        for ind, (act, pred) in enumerate(
            zip(output_dict_softmax[phase], predicted_dict[phase])
        ):

            if ind in rand_idx_dict[phase]:
                info_list.extend(
                    [[float(act_), float(pred_)]
                     for act_, pred_ in zip(act, pred)]
                )
        if os.path.exists(f"{phase}_output_correlation.csv"):
            os.remove(f"{phase}_output_correlation.csv")

        with open(f"{phase}_output_correlation.csv", "w") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerows(info_list)
