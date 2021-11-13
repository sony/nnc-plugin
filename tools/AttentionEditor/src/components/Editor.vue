<template>
  <div id="editorPane">
    <div id="editor">
      <canvas id="editorCanvas" />
      <img id="editorImage" v-on:load="onEditorImageLoad" />
    </div>
    <el-row>
      <el-col :span="4">
        <el-switch
          style="margin-right: 20px"
          v-model="cursorMode"
          active-text="Add"
          inactive-text="Remove"
        />
      </el-col>
      <el-col :span="4">Cursor size</el-col>
      <el-col :span="16">
        <el-slider
          v-model="cursorSize"
          show-input
          show-stops
          :min="cursorMin"
          :max="cursorMax"
          v-on:input="onEditorCursorSizeChange"
        />
      </el-col>
    </el-row>
    <el-row>
      <el-col :span="4">Zoom</el-col>
      <el-col :span="20">
        <el-slider
          v-model="zoomSize"
          show-input
          show-stops
          :min="zoomMin"
          :max="zoomMax"
          v-on:input="onEditorScaleChange"
        />
      </el-col>
    </el-row>
  </div>
</template>

<script>
import { defineComponent, onMounted, ref } from "vue";
import { useStore } from "vuex";

import * as fs from "fs";
import * as path from "path";
import * as PIXI from "pixi.js";
import debounce from "lodash";

export default defineComponent({
  name: "editor",
  props: {
    title: String,
  },
  setup() {
    const cursorMode = ref(true);
    const cursorSize = ref(5.0);
    const cursorMin = ref(1.0);
    const cursorMax = ref(100.0);
    const zoomSize = ref(0.0);
    const zoomMin = ref(0.0);
    const zoomMax = ref(20.0);

    const info = {
      app: null,
      dragging: false,
      cursor: null,
      imageSprite: null,
      mask: [
        {
          app: null,
          tex: null,
          spr: null,
          gra: null,
        },
        {
          app: null,
          tex: null,
          spr: null,
          gra: null,
        },
      ],
      editorWidth: -1,
      editorHeight: -1,
    };
    const store = useStore();

    function drawMap(inMapData) {
      const w = info.mask[1].app.renderer.width;
      const h = info.mask[1].app.renderer.height;
      const iW = inMapData[0].length;
      const iH = inMapData.length;
      const sW = Math.floor(w / iW);
      const sH = Math.floor(h / iH);
      info.mask.map((m) => {
        m.gra.clear();
      });
      for (let iY = 0; iY < iH; iY++) {
        for (let iX = 0; iX < iW; iX++) {
          const mask = inMapData[iY][iX];
          const c = (mask * 255) << 8;
          info.mask.map((m) => {
            m.gra
              .beginFill(c)
              .drawRect(iX * sW, iY * sH, sW, sH)
              .endFill();
          });
        }
      }
      info.mask.map((m) => {
        m.app.renderer.render(m.gra, m.tex, false);
      });
    }

    onMounted(() => {
      store.watch(
        () => store.state.requestMask,
        function (current) {
          if (current && info.app) {
            const maskData = {
              width: info.mask[1].app.renderer.width,
              height: info.mask[1].app.renderer.height,
              pixels: info.mask[1].app.renderer.plugins.extract.pixels(),
            };
            store.dispatch("outMaskData", maskData);
          }
        }
      );
      store.watch(
        () => store.state.inMaskData,
        function (current) {
          drawMap(current);
        }
      );
      store.watch(
        () => store.state.imageFilename,
        function (current) {
          const binary = fs.readFileSync(current);
          const base64data = new Buffer(binary).toString("base64");
          const type = path.extname(current).substring(1);
          const src = `data:image/${type};base64,${base64data}`;
          const image = document.getElementById("editorImage");
          if (image) {
            image.setAttribute("src", src);
          }
        }
      );
    });

    const redraw = function () {
      const div = document.getElementById("editor");
      const canvas = document.getElementById("editorCanvas");
      const image = document.getElementById("editorImage");
      if (div && canvas && image) {
        const iW = image.clientWidth;
        const iH = image.clientHeight;
        if (iW == 0 || iH == 0) {
          return;
        }
        const W = div.clientWidth;
        const H = div.clientHeight;
        const zW = Math.floor(W / iW);
        const zH = Math.floor(H / iH);

        zoomMin.value = 1;
        if (iW > iH) {
          cursorMax.value = iH;
        } else {
          cursorMax.value = iW;
        }
        if (cursorSize.value > cursorMax.value) {
          cursorSize.value = cursorMax.value;
        }
        if (zW > zH) {
          zoomMax.value = zH;
          if (zoomSize.value == 0) {
            zoomSize.value = zH;
          }
        } else {
          zoomMax.value = zW;
          if (zoomSize.value == 0) {
            zoomSize.value = zW;
          }
        }

        const sW = iW * zoomSize.value;
        const sH = iH * zoomSize.value;

        if (info.app) {
          const children = [];
          for (const child of info.app.stage.children) {
            children.push(child);
          }
          for (const child of children) {
            info.app.stage.removeChild(child);
          }
          info.cursor.destroy();
          info.cursor = null;
          info.imageSprite.destroy();
          info.imageSprite = null;
          info.mask.map((m) => {
            m.app.destroy();
            m.app = null;
            m.tex.destroy();
            m.tex = null;
            m.spr.destroy();
            m.spr = null;
            m.gra.destroy();
            m.gra = null;
          });
          info.app = null;
        }

        info.app = new PIXI.Application({
          resizeTo: canvas,
          antialias: true,
          backgroundAlpha: 0.2,
          view: canvas,
        });

        // Cursor
        info.cursor = new PIXI.Graphics();
        info.cursor.clear();
        info.cursor.lineStyle(1, 0xff00ff);
        info.cursor.drawCircle(0, 0, cursorSize.value * zoomSize.value);

        // Image
        info.imageSprite = PIXI.Sprite.from(image);
        info.imageSprite.width = sW;
        info.imageSprite.height = sH;
        info.imageSprite.x = (W - sW) / 2;
        info.imageSprite.y = (H - sH) / 2;

        // Mask
        info.mask[0].app = info.app;
        info.mask[0].tex = PIXI.RenderTexture.create(
          info.imageSprite.width,
          info.imageSprite.height
        );
        info.mask[0].spr = new PIXI.Sprite(info.mask[0].tex);
        info.mask[0].spr.width = iW;
        info.mask[0].spr.height = iH;
        info.mask[0].spr.setTransform(
          (W - sW) / 2,
          (H - sH) / 2,
          zoomSize.value,
          zoomSize.value
        );
        info.mask[0].spr.blendMode = PIXI.BLEND_MODES.ADD;
        info.mask[0].gra = new PIXI.Graphics();

        // Frame
        info.frame = new PIXI.Graphics();
        info.frame
          .clear()
          .beginFill(0xeeeeee)
          .drawRect(0, 0, W, (H - sH) / 2)
          .drawRect(0, 0, (W - sW) / 2, H)
          .drawRect((W + sW) / 2, 0, W, H)
          .drawRect(0, (H + sH) / 2, W, H)
          .endFill();

        info.app.stage.addChild(info.imageSprite);
        info.app.stage.addChild(info.mask[0].spr);
        info.app.stage.addChild(info.cursor);
        info.app.stage.addChild(info.frame);

        // Offline mask
        info.mask[1].app = new PIXI.Application({ width: iW, height: iH });
        info.mask[1].tex = PIXI.RenderTexture.create(iW, iH);
        info.mask[1].spr = new PIXI.Sprite(info.mask[1].tex);
        info.mask[1].spr.width = iW;
        info.mask[1].spr.height = iH;
        info.mask[1].gra = new PIXI.Graphics();

        info.mask[1].app.stage.addChild(info.mask[1].spr);

        info.imageSprite.interactive = true;

        info.app.stage.interactive = true;
        info.app.stage.hitArea = info.app.renderer.screen;

        if (info.maskdata) {
          info.mask.map((m) => {
            m.gra.clear();
          });
          const w = info.maskdata.width;
          const h = info.maskdata.height;
          for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
              const pos = ((h - y - 1) * w + x) * 4;
              const c =
                (info.maskdata.pixels[pos] << 16) |
                (info.maskdata.pixels[pos + 1] << 8) |
                info.maskdata.pixels[pos + 2];
              info.mask.map((m) => {
                m.gra.beginFill(c).drawRect(x, y, 1, 1).endFill();
              });
            }
          }
          info.mask.map((m) => {
            m.app.renderer.render(m.gra, m.tex, false);
          });
        }

        // Event handlers
        info.mask[1].app.renderer.on("postrender", () => {
          info.maskdata = {
            width: info.mask[1].app.renderer.width,
            height: info.mask[1].app.renderer.height,
            pixels: info.mask[1].app.renderer.plugins.extract.pixels(),
          };
        });

        // On click
        info.imageSprite.on("click", (e) => {
          console.log("click", e);
        });

        // On pointerdown
        // Drag start.
        info.imageSprite.on("pointerdown", () => {
          info.dragging = true;
        });

        // On pointerup
        // Drag end.
        info.imageSprite.on("pointerup", () => {
          info.dragging = false;
        });

        // On mousemove
        info.imageSprite.on("mousemove", (e) => {
          if (e.target) {
            const x = e.data.global.x - e.target.x;
            const y = e.data.global.y - e.target.y;
            const cX = Math.floor(x / zoomSize.value);
            const cY = Math.floor(y / zoomSize.value);

            info.cursor.position.set(e.data.global.x, e.data.global.y);

            if (info.dragging) {
              info.mask.map((m) => {
                let color = 0x00ff00;
                if (!cursorMode.value) {
                  color = 0x000000;
                }
                m.gra
                  .clear()
                  .beginFill(color)
                  .drawCircle(cX, cY, cursorSize.value)
                  .endFill();
                m.app.renderer.render(m.gra, m.tex, false);
              });
            }
          } else {
            info.dragging = false;
          }
        });
      }
    };

    const resize = function () {
      debounce(redraw(), 100);
    };

    window.onresize = resize;

    setInterval(function () {
      const div = document.getElementById("editor");
      if (div) {
        if (
          div.clientWidth != info.editorWidth ||
          div.clientHeight != info.editorHeight
        ) {
          info.editorWidth = div.clientWidth;
          info.editorHeight = div.clientHeight;
          resize();
        }
      }
    }, 500);

    return {
      cursorMode,
      cursorSize,
      cursorMin,
      cursorMax,
      zoomSize,
      zoomMin,
      zoomMax,
      onEditorImageLoad: () => {
        info.maskdata = null;
        redraw();
        if (store.state.imageFilename in store.state.attentionMaps) {
          drawMap(store.state.attentionMaps[store.state.imageFilename]);
        }
      },
      onEditorScaleChange: () => {
        resize();
      },
      onEditorCursorSizeChange: () => {
        if (info.cursor) {
          info.cursor.clear();
          info.cursor.lineStyle(1, 0xff00ff);
          info.cursor.drawCircle(0, 0, cursorSize.value * zoomSize.value);
        }
      },
    };
  },
});
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
#editorPane {
  margin: 1%;
  width: 98%;
  height: 100%;
}
#editor {
  width: 100%;
  height: 80%;
}
#editorImage {
  visibility: hidden;
}

#editorCanvas {
  width: 100%;
  height: 100%;
}
</style>
