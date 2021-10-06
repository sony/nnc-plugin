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
import sys
import numpy as np
import nnabla as nn
import nnabla.functions as F
from nnabla.utils.data_iterator import data_iterator
from tqdm import tqdm
from .model import resnet23_prediction, resnet56_prediction, loss_function
from .datasource import get_datasource
from .args import get_infl_args
par_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if par_dir not in sys.path:
    sys.path.append(par_dir)


CHECKPOINTS_PATH_FORMAT = "params_{}.h5"


def load_ckpt_path(checkpoint, train_epochs):
    checkpoints = [os.path.join(checkpoint, CHECKPOINTS_PATH_FORMAT.format(
        str(i))) for i in range(29, train_epochs, 30)]
    return checkpoints


def load_data(input_path):
    data_source = get_datasource(filename=input_path)
    num_iteration = data_source.size
    iterator = data_iterator(data_source, 1, None, False, False)
    return iterator, num_iteration


def calculate_ckpt_score(data_iterator, num_iteration, image_val, label_val, loss_val):
    ckpt_scores = []
    ds_idx_list = []
    img_path_list = []

    for i in range(num_iteration):
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


def get_scores(args, data_iterator, num_iteration, ckpt_paths):
    ckpt_scores = []
    ds_idx_list_ckpt = []
    img_path_list_ckpt = []

    if args.model == 'resnet23':
        model_prediction = resnet23_prediction
    elif args.model == 'resnet56':
        model_prediction = resnet56_prediction
    ncls = data_iterator._data_source.get_n_classes()
    image, *_ = data_iterator._data_source._get_data(0)
    image_shape = image.shape
    prediction = functools.partial(model_prediction,
                                   ncls=ncls,
                                   nmaps=64,
                                   act=F.relu,
                                   seed=args.seed)

    test = True
    image_val = nn.Variable(
        (1, image_shape[0], image_shape[1], image_shape[2]))
    label_val = nn.Variable((1, 1))
    pred_val, hidden = prediction(image_val, test)
    loss_val = loss_function(pred_val, label_val)

    for ckpt_path in tqdm(ckpt_paths):
        epoch = os.path.splitext(os.path.basename(ckpt_path))[0].split('_')[-1]
        nn.load_parameters(ckpt_path)

        ckpt_influences, ds_idx_list, img_path_list = calculate_ckpt_score(
            data_iterator, num_iteration, image_val, label_val, loss_val)
        if args.save_every_epoch:
            np.save(os.path.join(args.weight_output, (epoch + '_influence.npy')),
                    np.array([float(score.data) for score in ckpt_influences]))
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
    data_csv_path = os.path.join(args.weight_input, 'data_train.csv')
    data_source, num_iteration = load_data(data_csv_path)
    ckpt_paths = load_ckpt_path(args.checkpoint, args.train_epochs)

    results = get_scores(args, data_source, num_iteration, ckpt_paths)
    # sort by influence in ascending order
    rows = read_csv(data_csv_path)
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
