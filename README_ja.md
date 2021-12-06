# Neural Network Console プラグイン \([English](README.md)/日本語\)


![](./img/plugin.png)

## Neural Network Console プラグインとは？
プラグインはNeural Network Console上で前処理と後処理の機能追加を行うための仕組みです。

Neural Network Consoleにプラグインが同梱されていますが、このリポジトリ
から最新版のプラグインを追加したり、プラグインのソースコードを参考にし
て独自のプラグインを開発することができます。

## プラグインリスト
###  前処理プラグイン
* データセット作成
    * [Object detection (from Yolo v2 format)](./manuals/ja/Pre_Process/Create_Dataset/ObjectDetection.rst)
    * [Object detection (for CenterNet from Yolo v2 format)](./manuals/ja/Pre_Process/Create_Dataset/ObjectDetection_CenterNet.rst)
    * [Simple (Japanese) text classification](./manuals/ja/Pre_Process/Create_Dataset/SimpleTextClassification.rst)
    * [String classification](./manuals/ja/Pre_Process/Create_Dataset/StringClassification.rst)
    * [Simple tabular data](./manuals/ja/Pre_Process/Create_Dataset/SimpleTabularDataset.rst)
    * [Split image](./manuals/ja/Pre_Process/Create_Dataset/SplitImage.rst)
    * [Split wav](./manuals/ja/Pre_Process/Create_Dataset/SplitWav.rst)

###  後処理プラグイン
* ユーティリティ

    * [Cross tabulation](./manuals/ja/Post_Process/Utils/CrossTabulation.rst)
    * [CSV to wav](./manuals/ja/Post_Process/Utils/CSVtoWAV.rst)
    * [Inference](./manuals/ja/Post_Process/Utils/Inference.rst)
    * [Parameter stats](./manuals/ja/Post_Process/Utils/ParameterStats.rst)
    * [Restore split images / wav](./manuals/ja/Post_Process/Utils/RestoreSplitImageWav.rst)
    * [Similar words](./manuals/ja/Post_Process/Utils/SimilarWords.rst)
    * [Simple (Japanese) text generation](./manuals/ja/Post_Process/Utils/SimpleTextGeneration.rst)
    * [tSNE](./manuals/ja/Post_Process/Utils/tSNE.rst)
    
* 可視化
    * [Scatter plot](./manuals/ja/Post_Process/Visualization/ScatterPlot.rst)
    * [Tile images](./manuals/ja/Post_Process/Visualization/TileImages.rst)

* 説明可能なAI (XAI)
    * [SGD influence](./manuals/ja/Post_Process/XAI/SGDInfluence.rst)
    * [Influence Functions](./manuals/ja/Post_Process/XAI/InfluenceFunctions.rst)
    * [Face evaluation](./manuals/ja/Post_Process/XAI/FaceEvaluation.rst)
    * [Grad-CAM](./manuals/ja/Post_Process/XAI/GradCAM.rst)
    * [LIME](./manuals/ja/Post_Process/XAI/LIME.rst)
    * [SHAP](./manuals/ja/Post_Process/XAI/SHAP.rst)
    * [Smooth Grad](./manuals/ja/Post_Process/XAI/SmoothGrad.rst)
    * [TracIn](./manuals/ja/Post_Process/XAI/TracIn.rst)

## 最新のプラグインを利用するには

プラグインはNeural Network Consoleで動作します。もしNeural Network Consoleをお持ちでない場合には、こちら(https://dl.sony.com/)からダウンロードしてください。

1. このリポジトリからzipファイルをダウンロードします。
2. PC上でzipファイルを解凍します。
3. 既存のプラグインフォルダを削除します。neural_network_console>libs>**plugins **.にあります。
* **注意**いくつかのプラグインをオフにしたくない場合は、そのままにしておいてください。

4. ダウンロードした**plugins**フォルダをneural_network_console>libs>**plugins **.の同じ場所に置きます。

###  前処理プラグイン

* 前処理プラグインを実行するには、トップ画面左側の「DATASET」を選択し、「データセットの作成」をクリックします。そこで実行するプラグインを選択するこができます。

* To execute the plugins of the pre-processing, select the "DATASET" on the left of the top screen. Then  click "Create Dataset", you can select the plugins of the pre-processing.
<p align="center">
<img src="./img/Preprocessing.png" width="400px">  
</p>


### 後処理プラグイン

* 後処理プラグインを実行するには、 [評価] タブの評価結果を右クリックしてショートカットメニューを開き、実行するプラグインを選択します。
<p align="center">
<img src="./img/postprocessing.png" width="400px">  
</p>
