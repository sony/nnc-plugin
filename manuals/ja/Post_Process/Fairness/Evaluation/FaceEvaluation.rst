XAI/Face evaluation
~~~~~~~~~~~~~~~~~~~

ITA（Individual Typology Angle）と呼ばれるスコアを算出し、顔画像の肌の色を数値化します。

`Michele Merler, Nalini Ratha, Rogerio S Feris, John R Smith. "Diversity in faces". Computer Vision and Pattern Recognition, 2019. <https://arxiv.org/abs/1901.10436>`_

Input Information
===================

.. list-table::
   :widths: 30 70
   :class: longtable

   * - input
     - ITAスコアの算出を行う対象の画像ファイル一覧を含むデータセットCSVファイルを指定します

   * - output
     - 評価結果を出力するCSVファイルのファイル名を指定します


Output Information
===================

本プラグインの実行結果は 'output' で指定した名前のCSVファイルとして出力されます。
CSVファイル内の各カラムに関しての情報は以下の通りです（以下のリストに無い名称のカラムは、 '評価' の結果得られる output_result.csvと同様の意味です）。

.. list-table::
   :widths: 30 70
   :class: longtable

   * - ITA
     - 対象のインスタンスのITAスコアを表します。ITAスコアが高いほど肌の色が白く、低いほど肌の色が黒いことを示します。
