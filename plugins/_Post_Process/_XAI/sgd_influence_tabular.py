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
from sgd_influence_tabular_utils.network import get_config
from sgd_influence_tabular_utils.train import train
from sgd_influence_tabular_utils.infl import infl_sgd
from sgd_influence_utils.utils import delete_dir, get_context, ensure_dir, save_to_csv, calc_result_mean
from sgd_influence_tabular_utils.args import get_train_infl_args


def func(args):
    config = get_config(args)

    os.environ['NNABLA_CUDNN_DETERMINISTIC'] = '1'
    os.environ['NNABLA_CUDNN_ALGORITHM_BY_HEURISTIC'] = '1'

    # gpu/cpu
    ctx = get_context(config.device_id)
    nn.set_default_context(ctx)
    temp_dir = config.temp_dir
    ensure_dir(temp_dir)
    seeds = [i for i in range(args.n_trials)]
    base_infl_filename, ext = os.path.splitext(os.path.basename(args.output))
    infl_result_paths = []

    try:
        for seed in seeds:
            infl_result_path = os.path.join(
                temp_dir, f'{base_infl_filename}_{seed}{ext}')
            infl_result_paths.append(infl_result_path)
            config.infl_filepath = infl_result_path
            # train
            train(config, seed)
            # calc influence
            infl_sgd(config, seed)
        # calc average(calc_result_meanの実装)
        infl, header = calc_result_mean(infl_result_paths)
        # save average(save_to_csv)
        data_type = 'int,float,int'
        save_to_csv(filename=args.output, header=header,
                    list_to_save=infl, data_type=data_type)
        logger.log(99, 'SGD influence completed successfully.')
    except KeyboardInterrupt:
        pass
    # delete temporary files
    delete_dir(temp_dir, keyword='sgd_tabular_result')


def main():
    func(get_train_infl_args())


if __name__ == '__main__':
    main()
