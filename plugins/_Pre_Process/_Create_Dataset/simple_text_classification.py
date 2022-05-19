# Copyright 2021,2022 Sony Group Corporation.
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
import nltk

from simple_text_classification_util.simple_text_classification_util import create_dataset


def func(args):
    try:
        nltk.tokenize.word_tokenize("test")
    except LookupError:
        nltk.download('punkt')

    create_dataset(args, nltk.tokenize.word_tokenize)


def main():
    parser = argparse.ArgumentParser(
        description='Simple Text Classification\n\n' +
        'Convert a text classification dataset consisting of text x and label y to NNC format dataset.\n'
        'Use nltk.word_tokenize for text tokenization.\n\n',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-i',
        '--input',
        help='text classification dataset consisting of text x and label y (csv)',
        required=True)
    parser.add_argument(
        '-E',
        '--encoding',
        help='input text file encoding (text), default=utf-8-sig',
        default='utf-8-sig')
    parser.add_argument(
        '-l',
        '--max-length',
        help='maximum sentence length by the number of words (int) default=64',
        required=True,
        type=int,
        default=64)
    parser.add_argument(
        '-w',
        '--max-words',
        help='maximum number of words (int) default=256',
        required=True,
        type=int,
        default=256)
    parser.add_argument(
        '-m',
        '--min-occurrences',
        help='minimum number of word occurrences (int) default=10',
        type=int,
        default=10)
    parser.add_argument(
        '-n',
        '--normalize',
        help='normalize characters with unicodedata (bool) default=True',
        action='store_true')
    parser.add_argument(
        '-d', '--index-file-input', help='index file input (csv)')
    parser.add_argument(
        '-e',
        '--index-file-output',
        help='index file output (csv), default=index.csv')
    parser.add_argument(
        '-o', '--output-dir', help='output directory(dir)', required=True)
    parser.add_argument(
        '-g',
        '--log-file-output',
        help='log file output (file), default=log.txt')
    parser.add_argument(
        '-s',
        '--shuffle',
        help='shuffle (bool) default=True',
        action='store_true')
    parser.add_argument(
        '-f1',
        '--output_file1',
        help='output file name 1 (csv) default=train.csv',
        required=True,
        default='train.csv')
    parser.add_argument(
        '-r1',
        '--ratio1',
        help='output file 1 ratio as a percentage (int) default=100',
        type=float,
        required=True)
    parser.add_argument(
        '-f2',
        '--output_file2',
        help='output file name 2 (csv) default=test.csv',
        default='test.csv')
    parser.add_argument(
        '-r2',
        '--ratio2',
        help='output file 2 ratio as a percentage (int) default=0',
        type=float,
        default=0)
    parser.set_defaults(func=func)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
