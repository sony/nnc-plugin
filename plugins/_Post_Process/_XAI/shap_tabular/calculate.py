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

import numpy as np
from scipy.special import binom
import copy
import itertools
from nnabla import logger
import nnabla.utils.load as load


def check_datasize(data: np.array, std: int = 10):
    num_samples, num_features = data.shape
    m = 2**11 + 2 * num_features
    bites_per_element = 8
    estimated_size_gb = (num_samples * num_features * m) * \
        bites_per_element / (1024**3)
    return estimated_size_gb < std


class Ridge:
    def __init__(self, alpha=0, fit_intercept=True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept

    def fit(self, X: np.ndarray, y: np.ndarray, weights=None):
        y = y.reshape(-1, 1)
        assert X.shape[0] == y.shape[0], "Number of X and y rows don't match"

        if self.fit_intercept:
            X = np.concatenate([np.ones((X.shape[0], 1)), X], 1)

        if weights is None:
            tmp = X
        else:
            tmp = np.transpose(np.transpose(X) * np.transpose(weights))
        etmp_dot = np.dot(np.transpose(tmp), X)

        if self.alpha != 0:
            etmp_dot = etmp_dot + self.alpha * np.eye(etmp_dot.shape[0])

        try:
            tmp2 = np.linalg.inv(etmp_dot)
        except np.linalg.LinAlgError:
            tmp2 = np.linalg.pinv(etmp_dot)

        self.w = np.dot(tmp2, np.dot(np.transpose(tmp), y))

    def predict(self, X: np.ndarray):
        if self.fit_intercept:
            X = np.concatenate([np.ones((X.shape[0], 1)), X], 1)
        return np.dot(X, self.w)


class KernelSHAP:
    def __init__(self, data, model, X, alpha, nsamples='auto'):
        self.data = data
        self.model = model
        self.X = X
        self.alpha = alpha
        self.nsamples = nsamples

    def load_model(self, batch_size):
        class ForwardConfig:
            pass
        # Load model
        info = load.load(
            [self.model], prepare_data_iterator=False, batch_size=batch_size)

        config = ForwardConfig
        config.global_config = info.global_config

        config.executors = info.executors.values()

        config.networks = []
        if len(config.executors) < 1:
            logger.critical('Executor is not found in {}.'.format(self.model))
            return
        executor = list(config.executors)[0]
        if len(config.executors) > 1:
            logger.log(99, 'Only the first executor {} is used in the SHAP calculation.'.format(
                executor.name))

        if executor.network.name in info.networks.keys():
            config.networks.append(info.networks[executor.network.name])
        else:
            logger.critical('Network {} is not found in {}.'.format(
                executor.network.name, self.model))
            return

        # Prepare variable
        input_variable, data_name = list(executor.dataset_assign.items())[0]
        output_variable = list(executor.output_assign.keys())[0]
        return input_variable, output_variable

    def forward_model(self, input_variable, output_variable, inputs):
        # input image
        input_variable.variable_instance.d = inputs
        input_variable.variable_instance.need_grad = True

        # Forward
        output_variable.variable_instance.forward()
        return output_variable.variable_instance.d

    def shap_values(self):
        input_variable, output_variable = self.load_model(len(self.data))

        out = self.forward_model(input_variable, output_variable, self.data)

        self.train_samples = self.data.shape[0]
        self.num_features = self.data.shape[1]
        self.num_classes = out.shape[1]

        self.weights = np.ones(self.train_samples)
        self.weights /= np.sum(self.weights)
        self.nsamples_run = 0
        groups = [np.array([i]) for i in range(self.num_features)]
        self.fnull = np.sum((out.T * self.weights).T, 0)
        expected_value = self.fnull

        explanations = []
        for i in range(len(self.X)):
            self.index = i
            instance = self.X[i:i + 1, :]
            explanations.append(self.explain(instance, groups))
        s = explanations[0].shape

        outs = [np.zeros((self.X.shape[0], s[0])) for j in range(s[1])]
        for i in range(self.X.shape[0]):
            for j in range(s[1]):
                outs[j][i] = explanations[i][:, j]
        if len(self.X) == 1:
            outs = [sp[0] for sp in outs]

        return outs, expected_value

    def explain(self, instance, groups):
        self.nsamples_added = 0
        varying_inds = self.varying_groups(instance, groups)
        varying_feature_groups = [groups[i] for i in varying_inds]
        M = len(varying_feature_groups)
        varying_feature_groups = np.array(varying_feature_groups)
        varying_feature_groups = varying_feature_groups.flatten()

        if self.index == 0:
            self.input_variable, self.output_variable = self.load_model(1)
        model_out = self.forward_model(
            self.input_variable, self.output_variable, instance)
        fx = model_out[0]
        fx = np.array(fx)

        if M == 0:
            phi = np.zeros((self.num_features, self.num_classes))

        elif M == 1:
            phi = np.zeros((self.num_features, self.num_classes))
            diff = fx - self.fnull
            for d in range(self.num_classes):
                phi[varying_inds[0], d] = diff[d]

        else:
            if self.nsamples == "auto":
                self.nsamples = 2 * M + 2**11

            max_samples = 2 ** 30
            if M <= 30:
                max_samples = 2 ** M - 2
                if self.nsamples > max_samples:
                    self.nsamples = max_samples

            self.synth_data = np.tile(self.data, (self.nsamples, 1))
            self.mask_matrix = np.zeros((self.nsamples, M))
            self.kernel_weights = np.zeros(self.nsamples)
            self.y = np.zeros(
                (self.nsamples * self.train_samples, self.num_classes))
            self.ey = np.zeros((self.nsamples, self.num_classes))

            num_subset_sizes = np.int(np.ceil((M - 1) / 2.0))
            num_paired_subset_sizes = np.int(np.floor((M - 1) / 2.0))
            weight_vector = np.array([(M - 1.0) / (i * (M - i))
                                     for i in range(1, num_subset_sizes + 1)])
            weight_vector[:num_paired_subset_sizes] *= 2
            weight_vector /= np.sum(weight_vector)

            num_full_subsets = 0
            num_samples_left = self.nsamples
            group_inds = np.arange(M, dtype='int64')
            mask = np.zeros(M)
            remaining_weight_vector = copy.copy(weight_vector)

            for subset_size in range(1, num_subset_sizes + 1):

                nsubsets = binom(M, subset_size)
                if subset_size <= num_paired_subset_sizes:
                    nsubsets *= 2

                if num_samples_left * remaining_weight_vector[subset_size - 1] / nsubsets >= 1.0 - 1e-8:
                    num_full_subsets += 1
                    num_samples_left -= nsubsets

                    if remaining_weight_vector[subset_size - 1] < 1.0:
                        remaining_weight_vector /= (1 -
                                                    remaining_weight_vector[subset_size - 1])

                    w = weight_vector[subset_size - 1] / binom(M, subset_size)
                    if subset_size <= num_paired_subset_sizes:
                        w /= 2.0
                    for inds in itertools.combinations(group_inds, subset_size):
                        mask[:] = 0.0
                        mask[np.array(inds, dtype='int64')] = 1.0
                        self.addsample(varying_feature_groups,
                                       instance, mask, w)
                        if subset_size <= num_paired_subset_sizes:
                            mask[:] = np.abs(mask - 1)
                            self.addsample(
                                varying_feature_groups, instance, mask, w)
                else:
                    break

            self.run()

            phi = np.zeros((self.num_features, self.num_classes))
            for d in range(self.num_classes):
                vphi = self.solve(d, M, fx)
                phi[varying_inds, d] = vphi
        return phi

    def varying_groups(self, x, groups):
        varying = np.zeros(self.num_features)
        for i in range(0, self.num_features):
            inds = groups[i]
            x_group = x[0, inds]
            num_mismatches = np.sum(np.frompyfunc(
                self.not_equal, 2, 1)(x_group, self.data[:, inds]))
            varying[i] = num_mismatches > 0
        varying_indices = np.nonzero(varying)[0]
        return varying_indices

    def not_equal(self, i, j):
        if isinstance(i, str) or isinstance(j, str):
            return 0 if i == j else 1
        return 0 if np.isclose(i, j, equal_nan=True) else 1

    def addsample(self, varying_feature_groups, x, m, w):
        offset = self.nsamples_added * self.train_samples

        mask = m == 1.0
        groups = varying_feature_groups[mask]
        if len(groups.shape) == 2:
            for group in groups:
                self.synth_data[offset:offset +
                                self.train_samples, group] = x[0, group]
        else:
            evaluation_data = x[0, groups]
            self.synth_data[offset:offset +
                            self.train_samples, groups] = evaluation_data
        self.mask_matrix[self.nsamples_added, :] = m
        self.kernel_weights[self.nsamples_added] = w
        self.nsamples_added += 1

    def run(self):
        num_to_run = self.nsamples_added * self.train_samples - \
            self.nsamples_run * self.train_samples
        data_run = self.synth_data[self.nsamples_run *
                                   self.train_samples:self.nsamples_added*self.train_samples, :]
        if self.index == 0:
            self.input_variable_run, self.output_variable_run = self.load_model(
                num_to_run)
        modelout = self.forward_model(
            self.input_variable_run, self.output_variable_run, data_run)
        self.y[self.nsamples_run * self.train_samples:self.nsamples_added *
               self.train_samples, :] = np.reshape(modelout, (num_to_run, self.num_classes))

        # find the expected value of each output
        for i in range(self.nsamples_run, self.nsamples_added):
            ey_val = np.zeros(self.num_classes)
            for j in range(0, self.train_samples):
                ey_val += self.y[i * self.train_samples +
                                 j, :] * self.weights[j]

            self.ey[i, :] = ey_val

    def solve(self, dim, M, fx):
        ey_adj = self.ey[:, dim] - self.fnull[dim]
        nonzero_inds = np.arange(M)

        if len(nonzero_inds) == 0:
            return np.zeros(M)

        ey_adj2 = ey_adj - self.mask_matrix[:, nonzero_inds[-1]] * (
            fx[dim] - self.fnull[dim])
        etmp = np.transpose(np.transpose(
            self.mask_matrix[:, nonzero_inds[:-1]]) - self.mask_matrix[:, nonzero_inds[-1]])

        model = Ridge(alpha=self.alpha, fit_intercept=False)
        model.fit(etmp, ey_adj2, self.kernel_weights)
        w = model.w.reshape(-1)

        phi = np.zeros(M)
        phi[nonzero_inds[:-1]] = w
        phi[nonzero_inds[-1]] = (fx[dim] - self.fnull[dim]) - sum(w)

        for i in range(M):
            if np.abs(phi[i]) < 1e-10:
                phi[i] = 0

        return phi

    def calculate_shap(self):
        if len(self.X.shape) == 1:
            self.X = self.X.reshape((1, len(self.X)))
        values, expected_values = self.shap_values()
        return values, expected_values
