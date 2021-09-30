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
from nnabla import logger
from tracin_utils.args import get_train_infl_args
from tracin_utils.train import train
from tracin_utils.calculate_score import calc_infl
from tracin_utils.utils import delete_dir


def main():
    args = get_train_infl_args()
    os.environ['NNABLA_CUDNN_DETERMINISTIC'] = '1'
    os.environ['NNABLA_CUDNN_ALGORITHM_BY_HEURISTIC'] = '1'
    try:
        train(args)
        calc_infl(args)
        logger.log(99, 'TracIn completed successfully.')
    except KeyboardInterrupt:
        pass
    finally:
        delete_dir(args.model_save_path)


if __name__ == "__main__":
    main()
