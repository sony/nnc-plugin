#! /bin/bash
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
# Run this script to check if formatting is done, after running `make bwd-auto-format` at a clean nnabla-builder repository (submodules must be cloned).

FORMAT_CHECK_ERROR=0
git status | grep "modified:"
ERR=$?
if [ $ERR == 0 ]; then
    echo "Diff found" >&2
    mkdir -p output/format
    git diff > output/format/nnc-plugin.patch
    echo "A patch file is produced at \"output/format/nnc-plugin.patch\"" >&2
    FORMAT_CHECK_ERROR=1
else
    echo "No diff found"
fi
exit $FORMAT_CHECK_ERROR





