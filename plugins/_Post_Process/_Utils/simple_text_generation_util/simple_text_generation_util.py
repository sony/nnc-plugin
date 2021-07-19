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
import copy
import unicodedata

import heapq
from tqdm import tqdm
import numpy as np

from nnabla import logger
import nnabla.utils.load as load
from nnabla.utils.cli.utility import let_data_to_variable


def generate_text(args, tokenizer, join_char):
    # Load dictionary
    logger.log(99, 'Loading input dictionary ...')
    with open(args.index_file_input, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        dictionary = [row for row in reader]
    logger.log(99, f'  {len(dictionary)} words found in the dictionary.')
    dictionary_word = [x[1] for x in dictionary][2:]

    # Load model
    class ForwardConfig:
        pass

    logger.log(99, 'Loading model ...')
    info = load.load([args.model],
                     prepare_data_iterator=False,
                     batch_size=args.num_text)

    config = ForwardConfig
    config.global_config = info.global_config

    config.executors = info.executors.values()

    config.networks = []
    if len(config.executors) < 1:
        logger.critical('Executor is not found in {}.'.format(args.model))
        return
    executor = list(config.executors)[0]
    if len(config.executors) > 1:
        logger.log(
            99,
            '  Only the first executor {} is used for the text generation.'.format(
                executor.name))

    if executor.network.name in info.networks.keys():
        config.networks.append(info.networks[executor.network.name])
    else:
        logger.critical('Network {} is not found in {}.'.format(
            executor.network.name, args.model))
        return

    # Prepare variable
    inputs = {v: k for k, v in executor.dataset_assign.items()}
    input_variable_name = args.input_variable
    input_variable = inputs[input_variable_name]
    len_variable_name = args.length_variable
    len_variable = inputs[
        len_variable_name] if len_variable_name in inputs else None
    output_variable = list(executor.output_assign.keys())[0]
    if len(dictionary) != output_variable.variable_instance.d.shape[1]:
        logger.critical(
            f'The size of the dictionary and the model output do not match. {len(dictionary)}!={output_variable.variable_instance.d.shape[1]}.')
        return
    max_length = input_variable.variable_instance.d.shape[1]
    logger.log(99, f'  Maximum text length = {max_length}.')

    # Prepare seed text
    logger.log(99, 'Preparing for text generation ...')
    s = str(args.seed_text) if args.seed_text else ''
    if args.normalize:
        s = unicodedata.normalize('NFKC', s)
    # indexing
    s = list(tokenizer(s))[: max_length]
    s_in_index = [(dictionary_word.index(word) +
                   2 if word in dictionary_word else 1) for word in s]
    s_len = len(s_in_index)
    logger.log(99, f'  Seed string = {s}. Length = {s_len}.')
    s_in_index.extend([0] * (max_length - s_len))

    # Beam search
    x_in = np.tile(np.array(s_in_index), (args.num_text, 1))
    score_in = np.array([0.0] * args.num_text)
    finished = [False] * args.num_text
    x_num = 1
    for pos in tqdm(range(s_len, max_length)):
        # Input data
        let_data_to_variable(input_variable.variable_instance,
                             x_in,  # index sequence
                             data_name=input_variable_name, variable_name=input_variable.name)
        if len_variable:
            let_data_to_variable(len_variable.variable_instance,
                                 np.tile(
                                     np.array(pos), (args.num_text, 1)),  # len
                                 data_name=len_variable_name, variable_name=len_variable.name)
        # Generate data
        for v, generator in executor.generator_assign.items():
            v.variable_instance.d = generator(v.variable_instance.d.shape)
        # Forward
        executor.forward_target.forward(clear_buffer=True)

        # Generate
        if args.mode == 'beam-search':
            index_and_prob = []
            for i in range(x_num):
                index_and_prob.extend([[score_in[i] - np.log(x[1]), i, x[0]]
                                       for x in list(enumerate(list(output_variable.variable_instance.d[i])))])
            index_and_prob.sort(key=lambda x: x[0])

            x_out = copy.deepcopy(x_in)
            score_out = [0.0] * args.num_text
            for i in range(args.num_text):
                # previous word index sequence
                x_out[i] = x_in[index_and_prob[i][1]]
                x_out[i][pos] = index_and_prob[i][2]  # new word index
                finished[i] = finished[i] or index_and_prob[i][2] == 0
                score_out[i] = index_and_prob[i][0]
            x_in = x_out
            score_in = score_out
        elif args.mode == "sampling":
            for i in range(args.num_text):
                score = np.exp(
                    copy.deepcopy(
                        np.log(
                            output_variable.variable_instance.d[i])).astype(
                        np.float64) /
                    args.temperature)
                score = score / np.sum(score)
                word_index = np.argmax(np.random.multinomial(1, score, size=1))
                x_in[i][pos] = word_index
                score_in[i] = score_in[i] - \
                    np.log(output_variable.variable_instance.d[i][word_index])
                finished[i] = finished[i] or word_index == 0
        else:
            logger.critical(f'Mode "{args.mode}" is not supported.')
            return

        if all(finished):
            break
        x_num = args.num_text

    logger.log(99, 'Saving output file ...')
    with open(args.output, 'w', newline="\n", encoding='utf-8') as f:
        writer = csv.writer(f)
        x_in = list(x_in)
        for i in range(args.num_text):
            if 0 in list(x_in[i]):
                x_in[i] = x_in[i][:list(x_in[i]).index(0)]
            sequence = [idx for pos, idx in enumerate(
                x_in[i]) if idx > 0 and (pos == 0 or x_in[i][pos - 1] != idx)]
            s = join_char.join([dictionary[idx][1] for idx in sequence])
            print(score_in[i], s)
            writer.writerow([s])

    logger.log(99, 'Text generation completed successfully.')
