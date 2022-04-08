// Copyright 2022 Sony Group Corporation.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
import { createStore } from "vuex";

export default createStore({
  state: {
    imageFilename: null,
    requestMask: null,
    labelInfo: null,
    mapInfo: null,
    attentionMaps: null,
    outMaskData: null,
    inMaskData: null,
    serviceAddress: null,
  },
  mutations: {
    imageFilename(state, imageFilename) {
      state.imageFilename = imageFilename;
    },
    requestMask(state, requestMask) {
      state.requestMask = requestMask;
    },
    labelInfo(state, labelInfo) {
      state.labelInfo = labelInfo;
    },
    mapInfo(state, mapInfo) {
      state.mapInfo = mapInfo;
    },
    attentionMaps(state, attentionMaps) {
      state.attentionMaps = attentionMaps;
    },
    outMaskData(state, outMaskData) {
      state.outMaskData = outMaskData;
    },
    inMaskData(state, inMaskData) {
      state.inMaskData = inMaskData;
    },
    serviceAddress(state, serviceAddress) {
      state.serviceAddress = serviceAddress;
    },
  },
  actions: {
    imageFilename(context, imageFilename) {
      context.commit("imageFilename", imageFilename);
    },
    requestMask(context, requestMask) {
      context.commit("requestMask", requestMask);
    },
    labelInfo(context, labelInfo) {
      context.commit("labelInfo", labelInfo);
    },
    mapInfo(context, mapInfo) {
      context.commit("mapInfo", mapInfo);
    },
    attentionMaps(context, attentionMaps) {
      context.commit("attentionMaps", attentionMaps);
    },
    outMaskData(context, outMaskData) {
      context.commit("outMaskData", outMaskData);
    },
    inMaskData(context, inMaskData) {
      context.commit("inMaskData", inMaskData);
    },
    serviceAddress(context, serviceAddress) {
      context.commit("serviceAddress", serviceAddress);
    },
  },
  modules: {},
});
