# Copyright 2022,2023 Sony Group Corporation.
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

FROM node:16.19.1

ENV DEBIAN_FRONTEND=noninteractive

RUN npm install --global npm \
 && npm install --global --force yarn \
 && yarn global add @vue/cli

RUN dpkg --add-architecture i386 \
 && apt-get update \
 && apt-get install -y wine32 wine64 \
 && apt-get clean

CMD ["/bin/bash"]
