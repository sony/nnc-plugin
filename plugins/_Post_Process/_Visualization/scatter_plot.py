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
import plotly.graph_objs as go
from nnabla import logger


def extract_col(header, table, col_name):
    try:
        col_index = header.index(col_name)
    except BaseException:
        cols = [j for j, col in enumerate(header)if col_name == col.split('__')[
            0].split(':')[0]]
        if len(cols) == 0:
            logger.critical(
                'Variable {} is not found in the dataset.'.format(col_name))
            return []
        else:
            col_index = cols[0]
            logger.log(
                99, 'Column {} was used instead of column {}.'.format(
                    header[col_index], col_name))

    return [float(line[col_index]) for line in table]


def func(args):
    with open(args.input, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        table = [row for row in reader]

    variables = [args.x, args.y]
    if args.c:
        variables.append(args.c)
    cols = []
    for col_name in variables:
        col = extract_col(header, table, col_name)
        if len(col):
            cols.append(col)
        else:
            # Column not found
            return

    if args.c:
        data = [
            go.Scatter(
                x=cols[0],
                y=cols[1],
                mode='markers',
                marker={
                    'size': 12,
                    'color': cols[2],
                    'showscale':True})]
    else:
        data = [
            go.Scatter(
                x=cols[0],
                y=cols[1],
                mode='markers',
                marker={
                    'size': 12})]
    layout = go.Layout(
        xaxis={
            'title': args.x, 'gridcolor': '#c0c0c0', 'zerolinecolor': '#000000'}, yaxis={
            'title': args.y, 'gridcolor': '#c0c0c0', 'zerolinecolor': '#000000'}, width=int(
                args.width), height=int(
                    args.height), margin={
                        'l': 144, 'r': 144, 't': 144, 'b': 144, 'pad': 24, 'autoexpand': True}, font={
                            'size': 24}, plot_bgcolor='#FFFFFF')
    fig = go.Figure(data=data, layout=layout)

    fig.show()

    logger.log(99, 'Scatter plot completed successfully.')


def main():
    parser = argparse.ArgumentParser(
        description='Scatter plot\n\n' +
        'Draw a scatter plot based on the values in the two columns\n\n',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-i',
        '--input',
        help='path to input csv file (csv) default=output_result.csv',
        required=True,
        default='output_result.csv')
    parser.add_argument(
        '-x',
        '--x',
        help="column name for x axis (variable) default=y'",
        required=True,
        default="y’")
    parser.add_argument(
        '-y',
        '--y',
        help='column name for y axis (variable) default=y',
        required=True,
        default='y')
    parser.add_argument(
        '-c',
        '--c',
        help='column name for color label (variable) default=y',
        default=None)
    parser.add_argument(
        '-w', '--width', help="graph width (int) default=1920", default=1920)
    parser.add_argument(
        '-e', '--height', help="graph height (int) default=1080", default=1080)
    parser.set_defaults(func=func)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
