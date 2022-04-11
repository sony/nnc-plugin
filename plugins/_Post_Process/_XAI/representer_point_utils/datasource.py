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

'''
Provide data iterator for CIFAR10 examples.
'''
import numpy as np
import random
import os
import copy
from nnabla.utils.data_source_implements import CsvDataSource
from nnabla.utils.data_source_loader import _load_functions
from .utils import save_to_csv


class LabelShuffleCsvDataSource(CsvDataSource):
    def __init__(self, label_shuffle=False, **kwargs):
        super(LabelShuffleCsvDataSource, self).__init__(**kwargs)
        self._filepath_column_idx = self._get_column_idx('filepath')
        self._label_column_idx = self._get_column_idx('label')
        # self._shuffled_label_column = 'y_shuffled'
        self._shuffled_index_column = 'shuffled_index'
        self._label_shuffle_rate = 0.1
        # self._add_shuffled_label(label_shuffle)
        self._add_position()
        self._add_shuffled_index()
        self.reset()

    def _get_shuffled_label(self):
        num_cls = self.get_n_classes()
        raw_labels = np.array(self._get_labels()).astype(int).reshape(-1, 1)
        shuffled_labels = copy.deepcopy(raw_labels)
        extract_num = int(self._size * self._label_shuffle_rate // 10)
        for i in range(num_cls):
            extract_ind = np.where(raw_labels == i)[0]
            labels = [j for j in range(num_cls)]
            labels.remove(i)  # candidate of shuffle label
            artificial_label = [
                labels[int(i) % (num_cls - 1)] for i in range(int(extract_num))
            ]
            artificial_label = np.array(
                random.sample(artificial_label,
                              len(artificial_label))).astype('float32')
            convert_label = np.array([i for _ in range(len(extract_ind))])
            convert_label[-extract_num:] = artificial_label

            # change labels for the last several images
            shuffled_labels[extract_ind] = convert_label.reshape(-1, 1)
        return shuffled_labels

    def _update_variables(self, label, column_name):
        self._variables_dict[column_name] = {'label': label, 'value': None}
        self._columns.append((column_name, None, label))
        self._variables = tuple(self._variables_dict.keys())

    def _add_position(self):
        for position, row in enumerate(self._rows):
            row.append(str(position))
        label = 'datasource_index'
        self._update_variables(label, label)

    def _add_shuffled_index(self):
        for position, (row, shuffled_index) in enumerate(
                zip(self._rows, self._order)):
            row.append(str(shuffled_index))
        label = self._shuffled_index_column
        self._update_variables(label, label)

    def _add_shuffled_label(self, label_shuffle):
        _shuffled_label_column = self._shuffled_label_column
        if _shuffled_label_column not in self._variables_dict.keys():
            if label_shuffle:
                labels = self._get_shuffled_label()
            else:
                labels = np.array(self._get_labels()).astype(int).reshape(
                    -1, 1)
            for row, label in zip(self._rows, labels):
                row.append(str(label[0]))
            label = 'label_shuffled'
            self._update_variables(label, _shuffled_label_column)

    def _get_labels(self):
        return np.array(self._rows)[:, self._label_column_idx]

    def _get_column_idx(self, target):
        """
        Parameters
        ----------
        target: str
            'filepath' or 'label'

        Returns
        ----------
        i: int
            index of column that contains 'filepath' or 'label' info
        """
        if target not in ['filepath', 'label']:
            raise KeyError("target is 'filepath' or 'label'")

        def _is_filepath_idx(value):
            ext = (os.path.splitext(value)[1]).lower()
            if ext in _load_functions.keys():
                return True
            return False

        def _is_label_idx(value):
            try:
                value = float(value)
                return True
            except ValueError:
                return False

        func_dict = {'filepath': _is_filepath_idx, 'label': _is_label_idx}

        # judge from first row
        idx = 0
        for i, column_value in enumerate(self._rows[idx]):
            # Implemented refering to below
            # https://github.com/sony/nnabla/blob/f5eff2de5329ef02c40e7a5d7344abd91b19ece8/python/src/nnabla/utils/data_source_implements.py#L402
            # https://github.com/sony/nnabla/blob/f5eff2de5329ef02c40e7a5d7344abd91b19ece8/python/src/nnabla/utils/data_source_loader.py#L343
            if func_dict[target](column_value):
                return i
        raise RuntimeError('{} info is not in {}.'.format(
            target, self._filename))

    def _convert_to_abs_filepath(self):
        rows = copy.deepcopy(self._rows)
        for idx, row in enumerate(rows):
            file_name = self.get_abs_filepath_to_data(idx)
            row[self._filepath_column_idx] = file_name
        return rows

    def get_abs_filepath_to_data(self, idx):
        input_dir_name = os.path.abspath(os.path.dirname(self._filename))
        file_name = self.get_filepath_to_data(idx)
        file_name = os.path.join(input_dir_name, file_name)
        file_name = os.path.normpath(file_name)
        return file_name

    def get_filepath_to_data(self, idx):
        """
        Parameters
        ----------
        idx: int
            index of self._rows that includes filepath to data

        Returns
        ----------
        filepath: str
            ex: 'training/1.png'
        """
        return self._rows[idx][self._filepath_column_idx]

    def get_n_classes(self):
        """
        Parameters
        ----------
        None

        Returns
        ----------
        n_classes: int
            number of classes of label
            ex. n_classes is 10 if the dataset contains 0-9 labels 
        """

        labels = self._get_labels()
        n_classes = len(set(labels))
        return n_classes

    def save_to_csv(self,
                    filename,
                    header=None,
                    data_type=str,
                    convert_to_abs_path=True):
        if header is None:
            header = list(self._variables_dict.keys())
        if convert_to_abs_path:
            rows = self._convert_to_abs_filepath()
        else:
            rows = copy.deepcopy(self._rows)
        column_index = self.variables.index(self._shuffled_index_column)
        _rows = np.array(rows)
        _rows = _rows[_rows[:, column_index].astype(int).argsort()]
        save_to_csv(filename, header, _rows, data_type)


def get_datasource(filename,
                   shuffle=False,
                   label_shuffle=False,
                   normalize=False):
    get_datasource = LabelShuffleCsvDataSource(label_shuffle=label_shuffle,
                                               filename=filename,
                                               shuffle=shuffle,
                                               rng=None,
                                               normalize=normalize)
    return get_datasource
