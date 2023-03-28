Precision/Recall curve
~~~~~~~~~~~~~~~~~~~~~~~~~~

このプラグインは，Precision/Recall(PR)曲線を描画します．
PR曲線は様々なしきい値に対する精度と再現率のトレードオフを示す 2 次元プロットです．
PR曲線のAUCが高いほど，モデルが優れていることを示します．

.. list-table::
   :widths: 30 70
   :class: longtable

   * - input
     - データセットCSVファイルを指定します. PR曲線をプロットするための入力として使用されます．出力結果は"評価"タブに表示されます. デフォルトは `output_result.csv` が指定されます．

   * - target_variable
     - inputで指定したCSVファイルから，ターゲットラベルとして使用する列名を指定します．

   * - output_variable
     - inputで指定したCSVファイルから，モデル出力として使用する列名を指定します．

   * - width
     - 描画する散布図の横幅をインチ単位で指定します

   * - height
     - 描画する散布図の縦幅をインチ単位で指定します
