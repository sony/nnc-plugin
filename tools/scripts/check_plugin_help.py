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

import io
import os
import sys
import glob
import importlib
import threading

basedir = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..'))

ret = 0
for plugin in glob.glob(f'{basedir}/plugins/*/*/[A-Za-z]*.py'):
    plugin_dir = os.path.abspath(os.path.dirname(plugin))
    if plugin_dir not in sys.path:
        sys.path.append(plugin_dir)
    plugin_filename = os.path.basename(plugin)
    plugin_name, _ = os.path.splitext(plugin_filename)

    print(f'Checking {plugin_name}', file=sys.stderr)

    try:
        plugin_module = importlib.import_module(plugin_name)
    except:
        print(f'  Error could not import {plugin_filename}')
        ret += 1
        continue

    sys.argv = [plugin_name, '-h']
    old_stdout = sys.stdout
    sys.stdout = mystdout = io.StringIO()
    t = threading.Thread(target=plugin_module.main)
    t.start()
    t.join()

    sys.stdout = old_stdout
    if not mystdout.getvalue().startswith('usage'):
        print(f'  Error could not get help message from {plugin_filename}')
        ret += 1
    mystdout.close()

sys.exit(ret)
