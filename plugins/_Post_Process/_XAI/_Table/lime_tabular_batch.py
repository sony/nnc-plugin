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
import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
import csv
import collections
from sklearn.utils.extmath import softmax
import matplotlib.pyplot as plt
from nnabla import logger
import nnabla.utils.load as load
from nnabla.utils.cli.utility import let_data_to_variable
from nnabla.utils.data_source_implements import CsvDataSource


def get_mean_weight_lime(abs_mean):
    '''
    Mean weight plot for LIME weights
    '''
<<<<<<< HEAD
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
    y_ticks = range(len(abs_mean))
    y_labels = list(zip(*abs_mean))[0]
    plt.barh(y=y_ticks, width=list(zip(*abs_mean))[1])
    plt.yticks(ticks=y_ticks, labels=y_labels, size=8)
    for index, value in enumerate(list(zip(*abs_mean))[1]):
        plt.text(value, index, str(round(value, 4)), fontweight='bold')
    plt.title('Mean weight plot for LIME weights', fontweight='bold')
    plt.xlabel('Mean |Weight|', size=8, fontweight='bold')
=======
    fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(12,8))
    y_ticks = range(len(abs_mean))
    y_labels = list(zip(*abs_mean))[0]
    plt.barh(y=y_ticks,width=list(zip(*abs_mean))[1])
    plt.yticks(ticks=y_ticks,labels=y_labels,size= 8)
    for index, value in enumerate(list(zip(*abs_mean))[1]):
        plt.text(value, index, str(round(value,4)),fontweight='bold')
    plt.title('Mean weight plot for LIME weights',fontweight='bold')
    plt.xlabel('Mean |Weight|',size=8,fontweight='bold')
>>>>>>> 72bcd8e (plugin issues fixed)
    plt.savefig('Mean_LIME.png')


def get_beeswarm_lime(abs_mean, lime_weight, samples):
    '''
    Beeswarm plot for LIME weights
    '''
<<<<<<< HEAD
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 10))
    y_ticks = range(len(abs_mean))
    y_labels = list(zip(*abs_mean))[0]

    # plot scatterplot for each feature
    for i, feature in enumerate(y_labels):
        feature_weigth = lime_weight[:, i]
        feature_value = samples[:, i]
        plt.scatter(x=feature_weigth,
=======
    fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(15,10))
    y_ticks = range(len(abs_mean))
    y_labels = list(zip(*abs_mean))[0]

    #plot scatterplot for each feature
    for i,feature in enumerate(y_labels):
        feature_weigth = lime_weight[:,i]
        feature_value = samples[:,i]
        plt.scatter(x=feature_weigth ,
>>>>>>> 72bcd8e (plugin issues fixed)
                    y=[i]*len(feature_weigth),
                    c=feature_value,
                    cmap='bwr',
                    edgecolors='black',
<<<<<<< HEAD
                    alpha=0.8)
    plt.vlines(x=0, ymin=0, ymax=len(y_labels),
               colors='black', linestyles="--")
    plt.colorbar(label='Feature Value', ticks=[])
    plt.yticks(ticks=y_ticks, labels=y_labels, size=8)
    plt.xlabel('LIME Weight', size=8, fontweight='bold')
    plt.title('Beeswarm plot for LIME weights', fontweight='bold')
    plt.savefig('Beeswarm_LIME.png')


=======
                   alpha=0.8)
    plt.vlines(x=0,ymin=0,ymax=len(y_labels),colors='black',linestyles="--")
    plt.colorbar(label='Feature Value',ticks=[])
    plt.yticks(ticks=y_ticks,labels=y_labels,size=8)
    plt.xlabel('LIME Weight',size=8,fontweight='bold')
    plt.title('Beeswarm plot for LIME weights',fontweight='bold')
    plt.savefig('Beeswarm_LIME.png')
    
    
>>>>>>> 72bcd8e (plugin issues fixed)
def func(args):
    class ForwardConfig:
        pass
    # Load model
    info = load.load([args.model], prepare_data_iterator=False,
                     batch_size=args.num_samples)

    config = ForwardConfig
    config.global_config = info.global_config
    config.executors = info.executors.values()

    config.networks = []
    if len(config.executors) < 1:
        logger.critical('Executor is not found in {}.'.format(args.model))
        return
    executor = list(config.executors)[0]
    if len(config.executors) > 1:
        logger.log(99, 'Only the first executor {} is used in the LIMETABULAR calculation.'.format(
            executor.name))

    if executor.network.name in info.networks.keys():
        config.networks.append(info.networks[executor.network.name])
    else:
        logger.critical('Network {} is not found in {}.'.format(
            executor.network.name, args.model))
        return

    # Prepare variable
    input_variable, data_name = list(executor.dataset_assign.items())[0]
    output_variable = list(executor.output_assign.keys())[0]

    # Load csv
    d_input = CsvDataSource(args.input)
<<<<<<< HEAD
    required_column = [
        i for i in d_input._columns if i[0][0].casefold() == 'x']
=======
    required_column = [i for i in d_input._columns if i[0][0].casefold() == 'x']
>>>>>>> 72bcd8e (plugin issues fixed)
    index = []
    for col in required_column:
        index.append((d_input._columns).index(col))

    table = np.array([[float(r) for r in row] for row in d_input._rows])
<<<<<<< HEAD
    samples = table[:, index]
=======
    samples = table[:,index]
>>>>>>> 72bcd8e (plugin issues fixed)

    d_train = CsvDataSource(args.train)
    feature_names = []
    x = d_train.variables[0]
    for i, name in enumerate((d_train._variables_dict[x])):
        feature_name = '{}__{}:'.format(x, i) + name['label']
        feature_names.append(feature_name)
    train = np.array([[float(r) for r in row]
                     for row in d_train._rows])[:, :-1]

    results = []
    for sample in samples:
        # discretization
        to_discretize = list(set(range(train.shape[1])))
        discrete_train = train.copy()
        discrete_sample = sample.copy()
        freq = {}
        val = {}
        quartiles = {}
        quartile_boundary = {}
        quartile_mean = {}
        quartile_stds = {}

        for i in range(train.shape[1]):
            if i in to_discretize:
                column = train[:, i]
                quartile = np.unique(np.percentile(column, [25, 50, 75]))
                quartiles[i] = quartile
                discrete_train[:, i] = np.searchsorted(
                    quartile, column).astype(int)
                discrete_sample[i] = np.searchsorted(
                    quartile, discrete_sample[i]).astype(int)
                count = collections.Counter(discrete_train[:, i])
                val[i], f = map(list, zip(*(sorted(count.items()))))
                freq[i] = np.array(f) / np.sum(np.array(f))
                means = np.zeros(len(quartile) + 1)
                stds = np.zeros(len(quartile) + 1)
                for key in range(len(quartile) + 1):
                    tmp = column[discrete_train[:, i] == key]
                    means[key] = 0 if len(tmp) == 0 else np.mean(tmp)
                    stds[key] = 1.0e-11 if len(
                        tmp) == 0 else np.std(tmp) + 1.0e-11
                quartile_mean[i] = means
                quartile_stds[i] = stds
                quartile_boundary[i] = [
                    np.min(column)] + quartile.tolist() + [np.max(column)]
            else:
                count = collections.Counter(discrete_train[:, i])
                val[i], f = map(list, zip(*(sorted(count.items()))))
                freq[i] = np.array(f) / np.sum(np.array(f))

        discrete_data = np.zeros((args.num_samples, train.shape[1]))
        binary_data = np.zeros((args.num_samples, train.shape[1]))
        data_row = sample.copy()
        np.random.seed(0)
        for i in range(train.shape[1]):
            discrete_data[:, i] = np.random.choice(
                val[i], size=args.num_samples, replace=True, p=freq[i]).astype(int)
            binary_data[:, i] = (discrete_data[:, i] ==
                                 discrete_sample[i]).astype(int)

        discrete_data[0] = data_row

        binary_data[0] = np.ones_like(data_row)
        continuous_data = discrete_data.copy()
        discrete_data = discrete_data.astype(int)

        # undiscretization
        for i in to_discretize:
            mins = np.array(quartile_boundary[i])[discrete_data[1:, i]]
            maxs = np.array(quartile_boundary[i])[discrete_data[1:, i] + 1]
            means = np.array(quartile_mean[i])[discrete_data[1:, i]]
            stds = np.array(quartile_stds[i])[discrete_data[1:, i]]
            std_min = (mins - means) / stds
            std_max = (maxs - means) / stds
            unequal = (mins != maxs)

            ret = std_min
            ret[np.where(unequal)] = stats.truncnorm.rvs(
                std_min[unequal],
                std_max[unequal],
                loc=means[unequal],
                scale=stds[unequal]
            )
            continuous_data[1:, i] = ret
        continuous_data[0] = sample

        let_data_to_variable(input_variable.variable_instance, continuous_data,
                             data_name=data_name, variable_name=input_variable.name)

        # Forward
        executor.forward_target.forward(clear_buffer=True)

        pseudo_label = output_variable.variable_instance.d
        yss = softmax(pseudo_label)
<<<<<<< HEAD

=======
        
>>>>>>> 72bcd8e (plugin issues fixed)
        # regerssion
        def kernel(x, y):
            sigma = np.sqrt(train.shape[1]) * 0.75
            d = np.linalg.norm(y - x, axis=1)
            return np.sqrt(np.exp(-(d ** 2) / sigma**2))

        np.random.seed(0)
        weights = kernel(binary_data, binary_data[0])
        model = Ridge(alpha=1, fit_intercept=True)
        model.fit(binary_data, yss[:, args.class_index], sample_weight=weights)
        results.append(model.coef_)

    # Generate output csv
    with open(args.output, 'w', newline="\n") as f:
        writer = csv.writer(f)
        writer.writerow(
            ['Index'] + ['Importance of ' + x for x in feature_names])
        for i, result in enumerate(results):
            writer.writerow(
                [str(i + 1)] + ['{:.5f}'.format(value) for value in result])

    lime_csv = CsvDataSource(args.output)
    lime_weight = np.array([[float(r) for r in row]
<<<<<<< HEAD
                            for row in lime_csv._rows])
    lime_weight = lime_weight[:, 1:len(lime_csv._columns)]
    abs_mean = []
    abs_weight = []
    for i in range(len(lime_csv._columns)-1):
        abs_weight = np.absolute(lime_weight[:, i]).mean(axis=0)
        abs_mean.append([lime_csv._columns[i+1], abs_weight])

    # Get abs mean of LIME weights
    get_mean_weight_lime(abs_mean)

    # Get Beeswarm of LIME weights
    get_beeswarm_lime(abs_mean, lime_weight, samples)

    logger.log(99, 'Mean_weight plot and Beeswarm plot saved successfully.')
=======
                       for row in lime_csv._rows])
    lime_weight = lime_weight[:,1:len(lime_csv._columns)]
    abs_mean = []
    abs_weight = []
    for i in range(len(lime_csv._columns)-1):
        abs_weight = np.absolute(lime_weight[:,i]).mean(axis=0)
        abs_mean.append([lime_csv._columns[i+1],abs_weight])

    #Get abs mean of LIME weights
    get_mean_weight_lime(abs_mean)

    #Get Beeswarm of LIME weights
    get_beeswarm_lime(abs_mean, lime_weight, samples)
    
    logger.log(99, 'Mean_weight plot and Beeswarm plot saved successfully.')  
>>>>>>> 72bcd8e (plugin issues fixed)
    logger.log(99, 'LIME(tabular batch) completed successfully.')


def main():
    parser = argparse.ArgumentParser(
<<<<<<< HEAD
        description='LIME (all data)\n'
=======
        description='LIME tabular(all data)\n'
>>>>>>> 72bcd8e (plugin issues fixed)
                    '\n'
                    '"Why Should I Trust You?": Explaining the Predictions of Any Classifier\n' +
                    'Marco Tulio Ribeiro, Sameer Singh, Carlos Guestrin\n' +
                    'Knowledge Discovery and Data Mining, 2016.\n' +
                    'https://dl.acm.org/doi/abs/10.1145/2939672.2939778\n' +
                    '', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-m', '--model', help='path to model nnp file (model), default=results.nnp', required=True, default='results.nnp')
    parser.add_argument(
        '-i', '--input', help='path to input csv file (csv)', required=True)
    parser.add_argument(
        '-c2', '--class_index', help='class index (int), default=0', required=True, default=0, type=int)
    parser.add_argument(
        '-n', '--num_samples', help='number of samples (int), default=1000', required=True, default=1000, type=int)
    parser.add_argument(
        '-t', '--train', help='path to training dataset csv file (csv)', required=True)
    parser.add_argument(
        '-o', '--output', help='path to output csv file (csv), default=lime_tabular_batch.csv', required=True, default='lime_tabular_batch.csv')
    parser.set_defaults(func=func)
    args = parser.parse_args()

    args.func(args)


if __name__ == '__main__':
<<<<<<< HEAD
    main()
=======
    main()
>>>>>>> 72bcd8e (plugin issues fixed)
