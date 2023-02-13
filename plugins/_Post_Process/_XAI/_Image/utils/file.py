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
import csv
import numpy as np
from shutil import rmtree


def add_info_to_csv(rows, vals_to_add, column_name, position=-1):
    header = rows.pop(0)
    header.insert(position, column_name)
    for i, val_to_add in enumerate(vals_to_add):
        rows[i].insert(position, val_to_add)
    rows.insert(0, header)
    return rows


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


def delete_file(file_name):
    if os.path.isfile(file_name):
        os.remove(file_name)


def delete_dir(dir_name, keyword='sgd_infl_results'):
    if os.path.isdir(dir_name):
        if keyword in dir_name:
            rmtree(dir_name)


def ensure_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def save_to_csv(filename, header, list_to_save, data_type):
    with open(filename, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(header)
        writer.writerows(np.array([tuple(row)
                                   for row in list_to_save], dtype=data_type))


def read_csv(filename):
    with open(filename) as f:
        reader = csv.reader(f)
        ret = [s for s in reader]
    return ret
