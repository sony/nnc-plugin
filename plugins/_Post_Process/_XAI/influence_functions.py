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

import os
import nnabla as nn
from nnabla import logger
from sgd_influence_utils.model import get_config
from sgd_influence_utils.train import train
from sgd_influence_utils.utils import delete_dir, get_context, ensure_dir, save_to_csv, calc_result_mean
from sgd_influence_utils.args import get_train_infl_args_of_inflence_functions
from influence_functions_utils.infl import infl_icml


def func(args):
    alpha = args.alpha
    config = get_config(args)
    model_info_dict = config.model_info_dict
    file_dir_dict = config.file_dir_dict
    use_all_params = config.calc_infl_with_all_params
    need_evaluate = config.need_evaluate
    save_dir = args.weight_output

    os.environ['NNABLA_CUDNN_DETERMINISTIC'] = '1'
    os.environ['NNABLA_CUDNN_ALGORITHM_BY_HEURISTIC'] = '1'

    # gpu/cpu
    ctx = get_context(args.device_id)
    nn.set_default_context(ctx)
    ensure_dir(save_dir)
    seeds = [i for i in range(args.n_trials)]
    base_infl_filename, ext = os.path.splitext(file_dir_dict['infl_filename'])
    save_dir = file_dir_dict['save_dir']
    infl_result_paths = []

    try:
        for seed in seeds:
            infl_result_path = os.path.join(
                save_dir, f'{base_infl_filename}_{seed}{ext}')
            infl_result_paths.append(infl_result_path)
            file_dir_dict['infl_filename'] = infl_result_path
            model_info_dict['seed'] = seed
            # train
            train(model_info_dict, file_dir_dict,
                  use_all_params, need_evaluate)
            # calc influence
            infl_icml(model_info_dict, file_dir_dict,
                      use_all_params, need_evaluate, alpha)
        infl, header = calc_result_mean(infl_result_paths)
        # save
        data_type = 'object,int,float,int'
        save_to_csv(filename=f'{base_infl_filename}{ext}', header=header,
                    list_to_save=infl, data_type=data_type)
        logger.log(99, 'Influence functions completed successfully.')
    except KeyboardInterrupt:
        pass
    # delete temporary files
    delete_dir(save_dir, keyword='influence_functions_results')


def main():
    func(get_train_infl_args_of_inflence_functions())


if __name__ == '__main__':
    main()
