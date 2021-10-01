Influence Functions
~~~~~~~~~~~~~~~~~~~

Influence Functionsと呼ばれる方法を用いて、認識結果に対する入力画像の影響を評価します。
データセットとスコアは影響の大きい順に表示され、データ・クレンジングに参照できます。

Influence Functions経由でのブラックボックス予測については以下の論文を参照下さい。

Understanding Black-box Predictions via Influence Functions
  - Pang Wei Koh, Percy Liang
  - https://arxiv.org/abs/1703.04730

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

   * - seed
     - 入力列データをシャッフルするための乱数種を指定します。

   * - model
     - Influence functionsの計算で使用するモデルファイル (*.nnp) を指
       定します。[評価] タブで選択したトレーニング結果に基づいて
       Influence functionsを実行するには、既定のresults.nnnを使用します。

   * - batch_size
     - Influence functionsで使用するモデルを使用してトレーニングする
       バッチサイズを指定します。
