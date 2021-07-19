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
import argparse
import csv

import numpy as np

import nnabla as nn
import nnabla.utils.load as load
from nnabla import logger
from tqdm import tqdm


def func(args):
    # Load model
    info = load.load([args.model], prepare_data_iterator=False, batch_size=1)

    logger.log(99, 'Loading model file ...')
    with open(args.output, 'w', newline="\n") as f:
        writer = csv.writer(f)
        try:
            param = nn.get_parameters()[args.parameter].d
            logger.log(
                99, 'Loaded parameter {} of shape {} ...'.format(
                    args.parameter, param.shape))
        except BaseException:
            logger.critical(
                'Parameter "{}" is not found in the model file.'.format(
                    args.parameter))
            logger.critical(
                '"{}" contains the following parameters. {}'.format(
                    args.model, list(
                        nn.get_parameters().keys())))
            return

    logger.log(99, 'Loading input dictionary ...')
    with open(args.index_file_input, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        dictionary = [row for row in reader]
        logger.log(
            99, '{} words found in the dictionary.'.format(
                len(dictionary)))
        if len(dictionary) != param.shape[0]:
            logger.critical(
                'Dictionary and parameter sizes do not match. {} != {}'.format(
                    len(dictionary), param.shape[0]))
        dictionary_word = [x[1] for x in dictionary]
        if args.source_word in dictionary_word:
            word_index = dictionary_word.index(args.source_word)
        else:
            logger.critical(
                'Word "{}" is not found in the dictionary.'.format(
                    args.source_word))
            return

    def pearson(x, y):
        xd = x - np.mean(x)
        yd = y - np.mean(y)
        return np.dot(xd, yd) / (np.sqrt(sum(xd ** 2)) * np.sqrt(sum(yd ** 2)))

    logger.log(99, 'Calculating similarity ...')
    results = []
    for i, row in tqdm(enumerate(dictionary)):
        if i != word_index:
            # pearson(param[i], param[word_index])
            ip = np.dot(param[i], param[word_index])
            results.append([i, ip])

    results.sort(key=lambda x: x[1], reverse=True)

    with open(args.output, 'w', newline="\n", encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(
            ['Word', 'Inner Product with {}'.format(args.source_word)])
        for result in results[0:args.num_words]:
            logger.log(99, '{} : {}'.format(
                dictionary[result[0]][1], result[1]))
            writer.writerow([dictionary[result[0]][1], result[1]])

    logger.log(99, 'Similar Words completed successfully.')


def main():
    parser = argparse.ArgumentParser(
        description='Similar Words\n\nDisplay similar words based on the word-embedding.\n\n'
        '\n\n', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-m',
        '--model',
        help='path to model nnp file (model) default=results.nnp',
        required=True,
        default='results.nnp')
    parser.add_argument(
        '-p',
        '--parameter',
        help='parameter name of Embed layer (text) default=Embed/embed/W',
        required=True)
    parser.add_argument(
        '-d',
        '--index-file-input',
        help='index file input (csv)',
        required=True)
    parser.add_argument(
        '-w', '--source-word', help='source word (text)', required=True)
    parser.add_argument(
        '-n',
        '--num-words',
        help='number of similar words (int) default=10',
        required=True,
        type=int)
    parser.add_argument(
        '-o',
        '--output',
        help='path to output csv file (csv) default=similar_words.csv',
        required=True,
        default='similar_words.csv')
    parser.set_defaults(func=func)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
