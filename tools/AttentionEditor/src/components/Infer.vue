<template>
  <el-table
    :data="result"
    :default-sort="{ prop: 'value', order: 'descending' }"
    height="90vh"
    style="width: 100%"
  >
    <el-table-column prop="label" label="Label" />
    <el-table-column
      sortable
      prop="value"
      label="Value"
      :sort-method="sortValue"
    />
  </el-table>
  <el-button type="primary" size="small" @click="onInfer" style="width: 100%"
    >Infer</el-button
  >
  <el-dialog title="Warning" v-model="dialogVisible">
    <span>{{ dialogMessage }}</span>
  </el-dialog>
</template>

<script>
import { defineComponent, onMounted, reactive, toRefs } from "vue";
import { useStore } from "vuex";

import axios from "axios";

export default defineComponent({
  name: "Infer",
  setup() {
    const state = reactive({
      dialogMessage: "",
      dialogVisible: false,
      isInferencing: false,
      result: [],
      sortValue: (a, b) => {
        return parseFloat(a.value) - parseFloat(b.value);
      },
      onInfer: () => {
        if (state.isInferensing) {
          state.dialogMessage = "Another inference on going.";
          state.dialogVisible = true;
        } else {
          state.isInferencing = true;
          store.dispatch("requestMask", true);
        }
      },
    });
    const store = useStore();
    onMounted(() => {
      store.watch(
        () => store.state.outMaskData,
        function (current) {
          const mW = store.state.mapInfo.attention_map_shape[0];
          const mH = store.state.mapInfo.attention_map_shape[1];
          const sW = current.width / mW;
          const sH = current.height / mH;
          const maskData = new Array(mH);
          for (let y = 0; y < mH; y++) {
            maskData[y] = new Array(mW).fill(0);
            for (let x = 0; x < mW; x++) {
              let num = 0;
              let value = 0;
              for (let iY = 0; iY < sH; iY++) {
                for (let iX = 0; iX < sW; iX++) {
                  const x1 = Math.floor(x * sW + iX);
                  const y1 = Math.floor(current.height - 1 - y * sH - iY);

                  if (
                    x1 >= 0 &&
                    y1 >= 0 &&
                    x1 < current.width &&
                    y1 < current.height
                  ) {
                    const v =
                      current.pixels[(y1 * current.height + x1) * 4 + 1];
                    value += v;
                    num++;
                  }
                }
              }
              value = value / num;
              maskData[y][x] = value / 255.0;
            }
          }
          axios
            .get(`${store.state.serviceAddress}Infer`, {
              params: {
                map: JSON.stringify(maskData),
              },
            })
            .then((response) => {
              const result = response.data.result;
              state.result = [];
              for (let i = 0; i < result[0].length; i++) {
                let labelString = i;
                if (i in store.state.labelInfo) {
                  labelString = `${store.state.labelInfo[i]}: (${i})`;
                }
                state.result.push({
                  label: labelString,
                  value: result[0][i].toFixed(10),
                });
              }
              store.dispatch("requestMask", false);
              store.dispatch("inMaskData", response.data.att_map);
              state.isInferencing = false;
            })
            .catch((err) => {
              console.log("ERR", err);
            });
        }
      );
    });
    return toRefs(state);
  },
});
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
#textOver {
  text-overflow: ellipsis;
  overflow: hidden;
  white-space: nowrap;
}
</style>
