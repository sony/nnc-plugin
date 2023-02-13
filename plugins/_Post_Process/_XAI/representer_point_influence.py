# Copyright 2022 Sony Group Corporation.
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
from representer_point_utils.args import get_infl_args
from representer_point_utils.generate_feature import generate_feature
from representer_point_utils.compute_score import compute_score
from representer_point_utils.visualize import visuallize_infl


def main():
    args = get_infl_args()
    os.environ["NNABLA_CUDNN_DETERMINISTIC"] = "1"
    os.environ["NNABLA_CUDNN_ALGORITHM_BY_HEURISTIC"] = "1"
    try:
        generate_feature(args)
        info = os.path.join(args.monitor_path, "info.h5")

        print("## Compute Score")
        compute_score(args, info)
        print("## Saving Inflence samples")
        visuallize_infl(args, info)

        logger.log(99, "Representer Point completed successfully.")

    except KeyboardInterrupt:
        pass
    # finally:
    #     delete_dir(args.model_save_path)


if __name__ == "__main__":
    main()
