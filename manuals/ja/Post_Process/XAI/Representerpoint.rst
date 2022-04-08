XAI/RepresenterPointSelection
~~~~~~~~~
Representer Point Selectionと呼ばれる手法を用い，テストデータの予測において訓練データが与えた影響を算出します。
あるテストデータの推論において上位(または下位)k個の最も良い(悪い)影響を与えた訓練サンプルを提示します。
本プラグインはGPU利用時のみ動作します。

Representer Point Selection for Explaining Deep Neural Networks
   - Chih-Kuan Yeh, Joon Sik Kim, Ian E.H. Yen, Pradeep Ravikumar
   - https://proceedings.neurips.cc/paper/2018/file/8a7129b8f3edd95b7d969dfc2c8e9d9d-Paper.pdf

.. list-table::
   :widths: 30 70
   :class: longtable

   * - top_k
     - テストデータの推論において良/悪影響を与えた上位サンプルの提示する数を指定します

   * - num-samples
     - 影響を与えた訓練データを分析したいテストデータの数を指定します

   * - output
     - 評価結果を出力するCSVファイルのファイル名を指定します

   * - model
     - 事前学習を行ったモデルのパラメタ(nnp)を指定します

   * - input-train
     - 訓練に用いたデータセットを指定します。このデータセットから影響を与えたサンプルが提示されます

   * - input-val
     - テストデータセットを指定します。こちらのデータの推論において影響のあった訓練サンプルを提示します

   * - normalize
     - 前処理の正規化の有無を指定します

   * - lmbd
     - ファインチューニング時に必要となるL2正則化項の係数を指定します

   * - epoch
     - ファインチューニングのエポック数を指定します。


Output Information
===================

本プラグインの実行結果は 'output' で指定した名前のCSVファイルとして出力されます。 CSVファイル内の各カラムに関しての情報は以下の通りです

.. list-table::
   :widths: 30 70
   :class: longtable

   * - test_sample
     - テストサンプルの画像です

   * - label; pred
     - テストサンプルに対応するラベルと，予測ラベルです。

   * - positive_x
     - テストサンプルの推論において良い影響を与えた上位x番目の訓練サンプルを表示します。

   * - positive_x
     - テストサンプルの推論において悪い影響を与えた下位x番目の訓練サンプルを表示します。

Notes
===================
本スクリプトはクラス数分の出力次元数を持つAffine Layerを持つクラス分類ネットワークにおいてのみ有効です。
Sigmoidを利用するような回帰分析においては機能しません，


Link
========
| アルゴリズムに関する詳細な説明に関しましては以下のリンクをご参照ください。
| https://github.com/sony/nnabla-examples/tree/master/responsible_ai/representer_point#representer-point-selection

