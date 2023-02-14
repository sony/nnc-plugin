# Copyright 2021,2022,2023 Sony Group Corporation.
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

import concurrent.futures
import glob
import io
import os
import subprocess
import sys

from nnabla import logger

basedir = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..'))


def get_help(plugin):
    logger.critical(f'Getting help message of {os.path.basename(plugin)}')
    try:
        output = subprocess.check_output(['python3', plugin, '-h'])
    except:
        return []
    return output.decode().splitlines()


plugins = []
for plugin in sorted(glob.glob(f'{basedir}/plugins/_*/_*/[A-Za-z]*.py') +
                     glob.glob(f'{basedir}/plugins/_*/_*/_*/[A-Za-z]*.py')):
    plugins.append(plugin)

with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
    help_messages = executor.map(get_help, plugins)
ret = 0
for plugin, help_message in zip(plugins, help_messages):
    if len(help_message) == 0:
        print(f'    Error getting help from {os.path.basename(plugin)}')
        print(
            f'      To check what is wrong, please exec `python3 {plugin} -h` directly.')
        ret += 1
    else:
        print(
            f'    Help message ({os.path.basename(plugin)}): {help_message[0]}')
sys.exit(ret)
