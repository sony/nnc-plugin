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
import copy
import csv
import numpy as np
from nnabla import logger
from bias_mitigation_utils.utils import get_condition_vector


def save_info_to_csv(input_path, output_path, file_names, column_name='gradcam', insert_pos=0):
    with open(input_path, newline='') as f:
        rows = [row for row in csv.reader(f)]
    row0 = rows.pop(0)
    row0.insert(insert_pos, column_name)
    for i, file_name in enumerate(file_names):
        rows[i].insert(insert_pos, file_name)
    with open(output_path, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(row0)
        writer.writerows(rows)


def func(args):
    label_name = args.label_name
    protected_attribute = args.protected_attribute
    with open(args.input_train, 'r') as f:
        reader = csv.reader(f)
        col_names = next(reader)
        train = np.array([[float(r) for r in row] for row in reader])

    protected_attribute_names = protected_attribute.split(',')
    protected_attribute_indices = [col_names.index(
        s) for s in protected_attribute_names]

    # target value
    label_index = col_names.index(label_name)
    labels_train = train[:, label_index]

    # protected attributes
    df_prot = train[:, protected_attribute_indices]
    protected_attributes = copy.deepcopy(df_prot)
    privileged_groups = [{s: 1 for s in protected_attribute_names}]
    unprivileged_groups = [{s: 0 for s in protected_attribute_names}]

    favorable_label = 1.0  # good credit
    unfavorable_label = 0.0  # bad credit

    # equal weights for all classes by default in the train dataset
    instance_weights = np.ones_like(train[:, 0], dtype=np.float64)

    # get the only privileged condition vector for the given protected attributes
    # Values are `True` for the privileged values else 'False'
    privileged_cond = get_condition_vector(
        protected_attributes,
        protected_attribute_names,
        condition=privileged_groups)

    # Get the only unprivileged condition vector for the given protected attributes
    # Values are `True` for the unprivileged values else 'False)
    unprivileged_cond = get_condition_vector(
        protected_attributes,
        protected_attribute_names,
        condition=unprivileged_groups)

    # get the favorable(postive outcome) condition vector
    # Values are `True` for the favorable values else 'False'
    favorable_cond = labels_train.ravel() == favorable_label

    # get the unfavorable condition vector
    # Values are `True` for the unfavorable values else 'False'
    unfavorable_cond = labels_train.ravel() == unfavorable_label

    # combination of label and privileged/unprivileged groups

    # Postive outcome for privileged group
    privileged_favorable_cond = np.logical_and(favorable_cond, privileged_cond)

    # Negative outcome for privileged group
    privileged_unfavorable_cond = np.logical_and(
        unfavorable_cond, privileged_cond)

    # Postive outcome for unprivileged group
    unprivileged_favorable_cond = np.logical_and(
        favorable_cond, unprivileged_cond)

    # Negative outcome for unprivileged group
    unprivileged_unfavorable_cond = np.logical_and(
        unfavorable_cond, unprivileged_cond)

    instance_count = train.shape[0]
    privileged_count = np.sum(privileged_cond, dtype=np.float64)
    unprivileged_count = np.sum(unprivileged_cond, dtype=np.float64)
    favourable_count = np.sum(favorable_cond, dtype=np.float64)
    unfavourable_count = np.sum(unfavorable_cond, dtype=np.float64)

    privileged_favourable_count = np.sum(
        privileged_favorable_cond, dtype=np.float64)
    privileged_unfavourable_count = np.sum(
        privileged_unfavorable_cond, dtype=np.float64)
    unprivileged_favourable_count = np.sum(
        unprivileged_favorable_cond, dtype=np.float64)
    unprivileged_unfavourable_count = np.sum(
        unprivileged_unfavorable_cond, dtype=np.float64)

    # reweighing weights
    weight_privileged_favourable = favourable_count * \
        privileged_count / (instance_count * privileged_favourable_count)
    weight_privileged_unfavourable = unfavourable_count * \
        privileged_count / (instance_count * privileged_unfavourable_count)
    weight_unprivileged_favourable = favourable_count * \
        unprivileged_count / (instance_count * unprivileged_favourable_count)
    weight_unprivileged_unfavourable = unfavourable_count * \
        unprivileged_count / (instance_count * unprivileged_unfavourable_count)

    transformed_instance_weights = copy.deepcopy(instance_weights)
    transformed_instance_weights[privileged_favorable_cond] *= weight_privileged_favourable
    transformed_instance_weights[privileged_unfavorable_cond] *= weight_privileged_unfavourable
    transformed_instance_weights[unprivileged_favorable_cond] *= weight_unprivileged_favourable
    transformed_instance_weights[unprivileged_unfavorable_cond] *= weight_unprivileged_unfavourable

    save_info_to_csv(args.input_train, args.output,
                     transformed_instance_weights, column_name='Sample_weight')


def main():
    parser = argparse.ArgumentParser(
        description='reweighing\n' +
        '\n' +
        'Data preprocessing techniques for classification without discrimination\n' +
        'Kamiran, Faisal and Calders, Toon.\n' +
        'Knowledge and Information Systems, 33(1):1-33, 2012\n' +
        '', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-t', '--input-train', help='path to training dataset csv file (csv)', required=True)
    parser.add_argument(
        '-l', '--label-name', help='target label', required=True)
    parser.add_argument(
        '-p', '--protected_attribute', help='protected attribute', required=True)
    parser.add_argument(
        '-o', '--output', help='output file default=reweighing.csv', default='reweighing.csv')
    parser.set_defaults(func=func)

    args = parser.parse_args()

    args.func(args)
    logger.log(99, 'reweighing completed successfully.')


if __name__ == '__main__':
    main()
