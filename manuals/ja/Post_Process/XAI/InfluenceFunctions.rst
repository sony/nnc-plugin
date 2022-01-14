Influence Functions
~~~~~~~~~~~~~~~~~~~

Influence Functionsと呼ばれる方法を用いて、認識結果に対する入力画像の影響を評価します。
データセットとスコアは影響の大きい順に表示され、データ・クレンジングに参照できます。

Influence Functions経由でのブラックボックス予測については以下の論文を参照下さい。

`Pang Wei Koh, Percy Liang. "Understanding black-box predictions via influence functions". Proceedings of the 34th International Conference on Machine Learning, 2017 <http://proceedings.mlr.press/v70/koh17a>`_

Input Information
===================

.. list-table::
   :widths: 30 70
   :class: longtable
   :header-rows: 1

   * - Property
     - Notes

   * - input-train
     - Influence functions スコアが計算されるイメージファイルを含むデータセット
       CSVファイルを指定します。

   * - input-val
     - Influence functions スコアを計算する画像ファイルを含む
       データセットCSVファイルを指定します。
       この入力値データセットは入力列データセットに従ったInfluence functions
       スコア計算に使用されるが、スコアリングのターゲットは入力列データセット
       のみでとなります。入力系列以外のデータセットでCSVファイルを指定します。

   * - output
     - 推論結果を出力するCSVファイル名を指定します。

   * - n_trials
     - 試行回数を指定します。input-train で指定されたデータセットをシャッフルして指定した回数のInfluence計算を繰り返し、その平均値を算出するために用います。

   * - model
     - Influence functionsの計算で使用するモデルファイル (*.nnp) を指
       定します。[評価] タブで選択したトレーニング結果に基づいて
       Influence functionsを実行するには、既定のresults.nnnを使用します。

   * - batch_size
     - Influence functionsで使用するモデルを使用してトレーニングする
       バッチサイズを指定します。


Output Information
===================

本プラグインの実行結果は 'output' で指定した名前のCSVファイルとして出力されます。
CSVファイル内の各カラムに関しての情報は以下の通りです（以下のリストに無い名称のカラムは、 '評価' の結果得られる output_result.csvと同様の意味です）。

.. list-table::
   :widths: 30 70
   :class: longtable

   * - influence
     - 本プラグインによって計算された、対象のインスタンスのInfuence値です。本CSVファイルの行はこのInfluence値によってソートされています

   * - datasource_index
     - 対象のインスタンスの `input-train`のデータセットCSVファイルにおけるインデックスを意味します。再学習の際などに、 `input-train` のデータセットCSVファイルと同じ順番に並べ替える際に利用します
