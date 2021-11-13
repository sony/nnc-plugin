<template>
  <splitpanes class="default-theme" style="width: 100%; height: 100vh">
    <pane size="60%" min-size="30%"><Editor /></pane>
    <pane size="40%" min-size="10%">
      <Infer />
    </pane>
  </splitpanes>
</template>

<script>
import { defineComponent, onMounted, reactive, toRefs } from "vue";
import { Splitpanes, Pane } from "splitpanes";
import { useStore } from "vuex";

import axios from "axios";
import electron from "electron";

import Editor from "./components/Editor.vue";
import Infer from "./components/Infer.vue";
import "splitpanes/dist/splitpanes.css";

export default defineComponent({
  name: "App",
  components: {
    Editor,
    Infer,
    Splitpanes,
    Pane,
  },
  setup() {
    const state = reactive({
      serviceAddress: null,
      imageFilename: null,
    });

    const store = useStore();
    const ipcRenderer = electron.ipcRenderer;

    store.watch(
      () => store.state.serviceAddress,
      function (current) {
        console.log(`Service address is ${current}.`);

        axios
          .get(`${current}ImagePath`)
          .then((response) => {
            store.dispatch("imageFilename", response.data.path);
          })
          .catch((err) => {
            console.log("ERR", err);
          });

        axios
          .get(`${current}MapInfo`)
          .then((response) => {
            console.log("MapInfo", response.data);
            store.dispatch("mapInfo", response.data);
          })
          .catch((err) => {
            console.log("ERR", err);
          });

        axios
          .get(`${current}Labels`)
          .then((response) => {
            console.log("Labels", response.data);
            store.dispatch("labelInfo", response.data.labels);
          })
          .catch((err) => {
            console.log("ERR", err);
          });
      }
    );

    onMounted(async () => {
      async function getIpc(name) {
        const promise = new Promise((resolve) => {
          ipcRenderer.on(name, async (event, data) => {
            resolve(data);
          });
          ipcRenderer.send(name);
        });
        return promise;
      }
      state.serviceAddress = await getIpc("serviceAddress");
      store.dispatch("serviceAddress", state.serviceAddress);
    });
    return toRefs(state);
  },
});
</script>

<style>
body {
  overflow: hidden;
}

#app {
  width: 100%;
  height: 100vh;
  overflow: hidden;
}
</style>
