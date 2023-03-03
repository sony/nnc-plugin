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

.PHONY: all
all: win

.PHONY: clean
clean:
	rm -rf dist_electron

.PHONY: all-clean
all-clean: clean
	rm -rf node_modules

.PHONY: docker
docker:
	mkdir -p dist_electron/tmp
	echo electron-docker:$$(date +'%Y%m%d%H%M%S') > dist_electron/tmp/docker-image_id.txt
	docker build -t $$(cat dist_electron/tmp/docker-image_id.txt) docker |tee dist_electron/tmp/docker-build.log

.PHONY: yarn-docker
yarn-docker: docker
	docker run --rm -e HOME=$$(pwd)/dist_electron/tmp -u $$(id -u):$$(id -g) -v $$(pwd):$$(pwd) -w $$(pwd) $$(cat dist_electron/tmp/docker-image_id.txt) \
		yarn

.PHONY: win
win: yarn-docker
	docker run --rm -e HOME=$$(pwd)/dist_electron/tmp -u $$(id -u):$$(id -g) -v $$(pwd):$$(pwd) -w $$(pwd) $$(cat dist_electron/tmp/docker-image_id.txt) \
		yarn electron:build --win --x64 --dir

.PHONY: yarn
yarn:
	yarn

.PHONY: mac
mac: yarn
	yarn electron:build --mac --dir
