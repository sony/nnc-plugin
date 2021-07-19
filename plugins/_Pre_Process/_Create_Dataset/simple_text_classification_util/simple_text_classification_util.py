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
import os
import csv
import random
import logging
from tqdm import tqdm

from nnabla import logger

from collections import Counter
import unicodedata


def create_dataset(args, tokenizer):
    if args.log_file_output is not None:
        handler = logging.FileHandler(
            os.path.join(
                args.output_dir,
                args.log_file_output))
        logger.addHandler(handler)

    logger.log(99, 'Loading original dataset ...')
    with open(args.input, 'r', encoding=args.encoding) as f:
        reader = csv.reader(f)
        header = next(reader)
        table = [row for row in reader]

    if args.index_file_input is not None:
        logger.log(99, 'Loading input dictionary ...')
        with open(args.index_file_input, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            dictionary = [row for row in reader]
    else:
        logger.log(99, 'Creating dictionary ...')
        words = []
        for row in tqdm(table):
            s = row[0]
            if args.normalize:
                s = unicodedata.normalize('NFKC', s)
            words.extend(list(tokenizer(s))[: args.max_length])
        count = Counter(words).items()
        logger.log(
            99,
            '{} words are found in the input dataset.'.format(
                len(count)))
        count = [x for x in count if x[1] >= args.min_occurences]
        logger.log(
            99, '{} words have appeared more than {} times.'.format(
                len(count), args.min_occurences))
        count = list(sorted(count, key=lambda x: x[1], reverse=True))
        count = count[: args.max_words - 2]
        dictionary = [[0, '(EOS)'], [1, '(others)']]
        dictionary.extend([[i + 2, x[0]] for i, x in enumerate(count)])
        logger.log(
            99, '{} words were extracted for the dictionary.'.format(
                len(dictionary)))

    if args.index_file_output is not None:
        with open(os.path.join(args.output_dir, args.index_file_output), 'w', newline="\n", encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(dictionary)

    logger.log(99, 'Creating NNC dataset ...')
    dictionary_word = [x[1] for x in dictionary][2:]
    header.append('l')
    header.extend(['x__{}'.format(i) for i in range(args.max_length)])
    for row in tqdm(table):
        s = row[0]
        if args.normalize:
            s = unicodedata.normalize('NFKC', s)
        sentence = list(tokenizer(s))[: args.max_length]
        sentence_in_index = [
            (dictionary_word.index(word) +
             2 if word in dictionary_word else 1) for word in sentence]
        sentence_len = len(sentence_in_index)
        sentence_in_index.extend(
            [0] * (args.max_length - len(sentence_in_index)))
        row.append(sentence_len)
        row.extend(sentence_in_index)

    if args.shuffle:
        random.shuffle(table)

    header[0] = '# ' + header[0]
    logger.log(99, 'Saving output file 1 ...')
    with open(os.path.join(args.output_dir, args.output_file1), 'w', newline="\n", encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(table[: int(len(table) * args.ratio1) // 100])

    if args.output_file2 is not None and args.ratio2 > 0:
        logger.log(99, 'Saving output file 2 ...')
        with open(os.path.join(args.output_dir, args.output_file2), 'w', newline="\n", encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(table[int(len(table) * args.ratio1) // 100:])

    logger.log(99, 'Dataset creation completed successfully.')
