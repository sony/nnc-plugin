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
from nnabla import logger
from utils.file import save_info_to_csv
from gradcam_utils.gradcam_func import gradcam_batch_func


def main():
    parser = argparse.ArgumentParser(
        description='Grad-CAM (all data)\n' +
        '\n' +
        'Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization\n' +
        'Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, Dhruv Batra\n' +
        'Proceedings of the IEEE International Conference on Computer Vision, 2017.\n' +
        'https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html\n' +
        '', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-i', '--input', help='path to input csv file (csv) default=output_result.csv', required=True, default='output_result.csv')
    parser.add_argument(
        '-m', '--model', help='path to model nnp file (model) default=results.nnp', required=True, default='results.nnp')
    parser.add_argument(
        '-v1', '--input_variable', help='Variable to be processed (variable), default=x', required=True, default='x')
    parser.add_argument(
        '-ov', '--output_variable', help="output variable to visualize (variable) default=y0' ", required=True, default="y0'")
    parser.add_argument(
        '-v2', '--label_variable', help='Variable representing class index to visualize (variable) default=y', required=True, default='y')
    parser.add_argument(
        '-o', '--output', help='path to output csv file (csv) default=gradcam.csv', required=True, default='gradcam.csv')
    # designage if the model contains crop between input and first conv layer.
    parser.add_argument(
        '-cr', '--contains_crop', help=argparse.SUPPRESS, action='store_true')
    parser.set_defaults(func=gradcam_batch_func)

    args = parser.parse_args()

    file_names = args.func(args)

    save_info_to_csv(args.input, args.output,
                     file_names, column_name='gradcam')
    logger.log(99, 'Grad-CAM completed successfully.')


if __name__ == '__main__':
    main()
