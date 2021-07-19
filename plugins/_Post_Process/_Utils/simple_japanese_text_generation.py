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
import sys
import threading
import argparse
from janome.tokenizer import Tokenizer

from simple_text_generation_util.simple_text_generation_util import generate_text


def func(args):
    t = Tokenizer()

    def tokenizer(s):
        return t.tokenize(s, wakati=True)

    generate_text(args, tokenizer, '')


def main():
    parser = argparse.ArgumentParser(
        description='Simple Japanese Text Generation\n' +
        '\n' +
        'Generates a text based on a model that predicts the next word ' +
        'based on the word index series and its length' +
        '', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-m',
        '--model',
        help='path to model nnp file (model) default=results.nnp',
        required=True,
        default='results.nnp')
    parser.add_argument(
        '-v',
        '--input-variable',
        help='variable name for input data (variable) default=x',
        default='x')
    parser.add_argument(
        '-l',
        '--length-variable',
        help='variable name for text length (variable) default=l',
        default='l')
    parser.add_argument(
        '-d',
        '--index-file-input',
        help='index file input (csv)',
        required=True)
    parser.add_argument(
        '-s', '--seed-text', help='seed text (text), default=I am')
    parser.add_argument(
        '-n',
        '--normalize',
        help='normalize characters in seed text with unicodedata (bool) default=True',
        action='store_true')
    parser.add_argument(
        '-b',
        '--mode',
        help='mode (option:sampling,beam-search) default=sampling',
        default='sampling')
    parser.add_argument(
        '-t',
        '--temperature',
        help='temperature parameter for sampling mode(float), default=0.5',
        type=float,
        default=0.5)
    parser.add_argument(
        '-e',
        '--num-text',
        help='number of text to generate, beam-width for beam search (int), default=8',
        type=int,
        default=8)
    parser.add_argument(
        '-o',
        '--output',
        help='path to output image file (csv) default=text_generation.csv',
        required=True,
        default='text_generation.csv')
    parser.set_defaults(func=func)

    args = parser.parse_args()

    args.func(args)


if __name__ == '__main__':
    # thread.stack_size(8 * 1024 * 1024)
    sys.setrecursionlimit(1024 * 1024)
    main_thread = threading.Thread(target=main)
    main_thread.start()
    main_thread.join()
