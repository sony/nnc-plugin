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
import os
import shutil
from nnabla import logger
from utils.file import save_info_to_csv
from gradcam_utils.gradcam_func import gradcam_batch_func
from lime_utils.lime_func import lime_batch_func
from shap_utils.shap_func import shap_batch_func
from smoothgrad_utils.smoothgrad_func import smoothgrad_batch_func


def func(args):
    if (not args.GradCAM) and (not args.LIME) and (not args.SHAP) and (not args.SmoothGrad):
        logger.critical('Task to run is not found.')
        raise RuntimeError(
            "At least one task (Grad-CAM, LIME, SHAP, SmoothGrad) is necessary to use this plugin.")

    shutil.copyfile(args.input, args.output)

    data_output_dir = os.path.splitext(args.output)[0]

    class Args:
        pass

    index_count = 1

    if args.GradCAM:
        gradcam_args = Args()
        gradcam_args.input = args.input
        gradcam_args.model = args.model
        gradcam_args.input_variable = args.input_variable
        gradcam_args.label_variable = args.label_variable
        gradcam_args.output = data_output_dir + '_' + 'gradcam.csv'
        gradcam_args.contains_crop = args.contains_crop
        gradcam_file_names = gradcam_batch_func(gradcam_args)
        save_info_to_csv(args.output, args.output,
                         gradcam_file_names, column_name='Grad-CAM', insert_pos=index_count)
        index_count += 1

    if args.SmoothGrad:
        smoothgrad_args = Args()
        smoothgrad_args.model = args.model
        smoothgrad_args.noise_level = args.noise_level
        smoothgrad_args.num_samples = args.num_samples_smoothgrad
        smoothgrad_args.layer_index = args.layer_index
        smoothgrad_args.input = args.input
        smoothgrad_args.output = data_output_dir + '_' + 'smoothgrad.csv'
        smoothgrad_args.input_variable = args.input_variable
        smoothgrad_args.label_variable = args.label_variable
        smoothgrad_file_names = smoothgrad_batch_func(smoothgrad_args)
        save_info_to_csv(args.output, args.output,
                         smoothgrad_file_names, column_name='SmoothGrad', insert_pos=index_count)
        index_count += 1

    if args.LIME:
        lime_args = Args()
        lime_args.input = args.input
        lime_args.model = args.model
        lime_args.input_variable = args.input_variable
        lime_args.label_variable = args.label_variable
        lime_args.num_samples = args.num_samples_lime
        lime_args.num_segments = args.num_segments
        lime_args.num_segments_2 = args.num_segments_2
        lime_args.output = data_output_dir + '_' + 'lime.csv'
        lime_file_names = lime_batch_func(lime_args)
        save_info_to_csv(args.output, args.output,
                         lime_file_names, column_name='LIME', insert_pos=index_count)
        index_count += 1

    if args.SHAP:
        shap_args = Args()
        shap_args.input = args.input
        shap_args.model = args.model
        shap_args.input_variable = args.input_variable
        shap_args.label_variable = args.label_variable
        shap_args.num_samples = args.num_samples_shap
        shap_args.batch_size = args.batch_size
        shap_args.interim_layer = args.interim_layer
        shap_args.output = data_output_dir + '_' + 'shap.csv'
        shap_file_names = shap_batch_func(shap_args)
        save_info_to_csv(args.output, args.output,
                         shap_file_names, column_name='SHAP', insert_pos=index_count)
        index_count += 1


def main():
    parser = argparse.ArgumentParser(
        description='XAI Visualization (all data) (GradCAM, SHAP, LIME, SmoothGrad)\n' +
        'Grad-CAM\n' +
        '\n' +
        'Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization\n' +
        'Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, Dhruv Batra\n' +
        'Proceedings of the IEEE International Conference on Computer Vision, 2017.\n' +
        'https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html\n' +
        '' +
        'LIME (image)\n' +
        '\n' +
        '"Why Should I Trust You?": Explaining the Predictions of Any Classifier\n' +
        'Marco Tulio Ribeiro, Sameer Singh, Carlos Guestrin\n' +
        'Knowledge Discovery and Data Mining, 2016.\n' +
        'https://dl.acm.org/doi/abs/10.1145/2939672.2939778\n' +
        '' +
        'SHAP\n'
        '\n'
        'A Unified Approach to Interpreting Model Predictions\n'
        'Scott Lundberg, Su-In Lee\n' +
        'Proceedings of the 31st International Conference on Neural Information Processing Systems, 2017.\n' +
        'https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html\n' +
        '' +
        'SmoothGrad\n' +
        '\n' +
        'SmoothGrad: removing noise by adding noise\n' +
        'Daniel Smilkov, Nikhil Thorat, Been Kim, Fernanda Viegas, Martin Wattenberg\n' +
        'Workshop on Visualization for Deep Learning, ICML, 2017.\n' +
        'https://arxiv.org/abs/1706.03825\n' +
        '', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-i', '--input', help='path to input csv file (csv) default=output_result.csv', required=True, default='output_result.csv')
    parser.add_argument(
        '-m', '--model', help='path to model nnp file (model) default=results.nnp', required=True, default='results.nnp')
    parser.add_argument(
        '-v1', '--input_variable', help='Variable to be processed (variable), default=x', required=True, default='x')
    parser.add_argument(
        '-v2', '--label_variable', help='Variable representing class index to visualize (variable) default=y', required=True, default='y')
    parser.add_argument(
        '-o', '--output', help='path to output csv file (csv) default=xai_visualization.csv', required=True, default='xai_visualization_batch.csv')
    # designage if the model contains crop between input and first conv layer.
    parser.add_argument(
        '-cr', '--contains_crop', help=argparse.SUPPRESS, action='store_true')
    parser.add_argument(
        '-n_lime', '--num_samples_lime', help='(LIME) number of samples N (int), default=1000', required=True, type=int, default=1000)
    parser.add_argument(
        '-n_shap', '--num_samples_shap', help='(SHAP) number of samples N (int), default=100', required=True, type=int, default=100)
    parser.add_argument(
        '-n_smoothgrad', '--num_samples_smoothgrad', help='(SmoothGrad) number of samples N (int), default=25', required=True, type=int, default=25)
    parser.add_argument(
        '-s', '--num_segments', help='(LIME) number of segments (int), default=10', required=True, type=int, default=10)
    parser.add_argument(
        '-s2', '--num_segments_2', help='(LIME) number of segments to highlight (int), default=3', required=True, type=int, default=3)
    parser.add_argument(
        '-b', '--batch_size', help='(SHAP) batch size, default=50', required=True, type=int, default=50)
    parser.add_argument(
        '-il', '--interim_layer', help='(SHAP) layer input to explain, default=0', required=True, type=int, default=0)
    parser.add_argument(
        '-nl', '--noise_level', help='(SmoothGrad) noise level(0.0 to 1.0) to calculate standard deviation for input image, default=0.15', type=float, default=0.15)
    # index of layer of interest to visualize (input layer is 0), default=0
    parser.add_argument(
        '-li', '--layer_index', help=argparse.SUPPRESS, type=int, default=0)
    parser.add_argument(
        '-sm', '--SmoothGrad', action='store_true', help='SmoothGrad (bool), default=True')
    parser.add_argument(
        '-g', '--GradCAM', action='store_true', help='Grad-CAM (bool), default=True')
    parser.add_argument(
        '-l', '--LIME', action='store_true', help='LIME(image) (bool), default=False')
    parser.add_argument(
        '-sh', '--SHAP', action='store_true', help='SHAP (bool), default=False')

    args = parser.parse_args()
    func(args)
    logger.log(
        99, 'XAI Visualization (batch) (GradCAM, SHAP, LIME, SmoothGrad) (image) completed successfully.')


if __name__ == '__main__':
    main()
