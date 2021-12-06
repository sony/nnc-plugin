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
from utils.model import get_context
from utils.file import read_csv, add_info_to_csv, save_to_csv
import functools
import os
from nnabla import logger
import sys
import numpy as np
import nnabla as nn
import nnabla.functions as F
from nnabla.utils.nnp_graph import NnpLoader
import nnabla.utils.load as load
from nnabla.utils.data_iterator import data_iterator
from tqdm import tqdm
from .model import resnet23_prediction, resnet56_prediction
from .datasource import get_datasource
from .args import get_infl_args

par_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if par_dir not in sys.path:
    sys.path.append(par_dir)

def categorical_cross_entropy(softmax, label):
    loss = F.mean(F.categorical_cross_entropy(softmax, label))
    return loss


def load_data(input_path, normalize):
    data_source = get_datasource(filename=input_path, normalize=normalize)
    num_iteration = data_source.size
    iterator = data_iterator(data_source, 1, None, False, False)
    return iterator, num_iteration


def calculate_ckpt_score(data_iterator, num_iteration, image_val, label_val, loss_val, softmax):
    ckpt_scores = []
    ds_idx_list = []
    img_path_list = []

    for i in tqdm(range(num_iteration)):
        inputs = data_iterator.next()
        for name, param in nn.get_parameters().items():
            param.grad.zero()  # grad initialize
            if 'affine' not in name:
                param.need_grad = False

        grads = []
        images, labels, *_extra_info = inputs
        shuffled_labels, ds_idxes, _shuffled_idxes = _extra_info
        image_val.d, label_val.d = images, shuffled_labels

        loss_val.forward()
        loss_val.backward()

        for name, param in nn.get_parameters().items():
            if 'affine' in name:
                grads.append(param.grad)
        grad_mul = [F.sum(grad * grad) for grad in grads]
        score = F.add_n(*grad_mul)
        ckpt_scores.append(score)
        ds_idx = ds_idxes.astype(int)[0, 0]
        ds_idx_list.append(ds_idx)
        abs_img_path = data_iterator._data_source.get_abs_filepath_to_data(
            ds_idx)
        img_path_list.append(abs_img_path)
    return ckpt_scores, ds_idx_list, img_path_list


def check_sum_direction(ds_idx_list_ckpt):
    # summation axis must have data from same image
    for i, ds_idx_list in enumerate(ds_idx_list_ckpt):
        if i == 0:
            first_ds_idx_list = ds_idx_list
        else:
            is_same_idx = first_ds_idx_list == ds_idx_list
            if is_same_idx:
                pass
            else:
                raise Exception("Summation of data for different image source")


def get_scores(args, data_iterator, num_iteration):
    ckpt_scores = []
    ds_idx_list_ckpt = []
    img_path_list_ckpt = []

    nnp = NnpLoader(args.model_path)
    model = nnp.get_network('Runtime', 1)
    image_val = model.inputs["Input"]
    label_val = nn.Variable((1, 1))
    softmax = model.outputs["y'"]
    loss_val = categorical_cross_entropy(softmax, label_val)
    
    ckpt_influences, ds_idx_list, img_path_list = calculate_ckpt_score(
        data_iterator, num_iteration, image_val, label_val, loss_val, softmax)

    ckpt_scores.append(ckpt_influences)
    ds_idx_list_ckpt.append(ds_idx_list)
    img_path_list_ckpt.append(img_path_list)
    sum_ckpt_scores = []
    for ind in range(num_iteration):
        tmp = 0
        for ckpt_score in ckpt_scores:
            tmp += float(ckpt_score[ind].data)
        sum_ckpt_scores.append(tmp)
    check_sum_direction(ds_idx_list_ckpt)
    return {
        'img_path': img_path_list_ckpt[0],
        'influence': sum_ckpt_scores,
        'datasource_index': ds_idx_list_ckpt[0],
    }


def calc_infl(args):
    ctx = get_context(device_id=args.device_id)
    nn.set_default_context(ctx)
    data_source, num_iteration = load_data(args.input_train, args.normalize)

    results = get_scores(args, data_source, num_iteration)
    
    # sort by influence in ascendissng order
    rows = read_csv(args.input_train)
    # print(rows)
    rows = [r[:-3] + [r[-2]] for r in rows]
    rows = add_info_to_csv(rows, results['influence'], 'influence', position=0)
    header = rows.pop(0)
    rows = np.array(rows)
    rows = rows[rows[:, 0].astype(float).argsort()[::-1]]
    save_to_csv(filename=args.output, header=header,
                list_to_save=rows, data_type=str)


if __name__ == "__main__":
    args = get_infl_args()
    calc_infl(args)
