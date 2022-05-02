XAI/Attention Map Visualization
~~~~~~~~~~~~~~~~~~~~~

画像分類を行うモデルにおいて、Attention Branch NetworkのAttention Mapを入力画像に重ねて可視化します。

| Attention Branch Network:
| `Hiroshi Fukui, Tsubasa Hirakawa, Takayoshi Yamashita, Hironobu Fujiyoshi. "Attention Branch Network: Learning of Attention Mechanism for Visual Explanation". 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). <https://ieeexplore.ieee.org/document/8953929>`_

Input Information
===================

.. list-table::
   :widths: 30 70
   :class: longtable

   * - model
     -
        Attention Map Visualizationの演算に用いるのモデルファイル（*.nnp）を指定します。
        
        評価タブで選択中の学習結果を元にAttention Map Visualization を行うには、デフォルトのresults.nnpを使用します。

   * - image
     -
        分析を行う画像ファイルを指定します。
        
        評価タブの評価結果で表示されている特定の画像に対してAttention Map Visualization を行うには、画像ファイル名の係れたセルが選択された状態でプラグインを起動することで、imageに画像ファイル名が自動入力されます。

   * - output
     -
        可視化結果を出力するデータセットCSVファイルのファイル名を指定します。
        
        評価タブの評価結果からAttention Map Visualization を実行した場合、学習結果フォルダに指定したファイル名で可視化結果が保存されます。

   * - attention_map_layer
     -
        Attention Branch Networkのモデルを構成する層の内、Attention Mapの層の名前を指定します。


Output Information
===================

本プラグインの実行結果は 'output' で指定した名前のPNGファイルとして出力されます。
対象のインスタンスへの本プラグインの適用結果が表示されます。 元画像にjet カラーマップを重ねた形で表示され、赤色に近いほど分類結果に強く影響した箇所であることを示します。


XAI/Attention Map Visualization(batch)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

画像分類を行うモデルにおいて、Attention Branch NetworkのAttention Mapを入力画像に重ねて可視化します。
XAI/Attention Branch Network(Attention Map Visualization)プラグインが1枚の画像に対して処理を行うのに対し、XAI/Attention Branch Network(Attention Map Visualization batch)プラグインは指定するデータセットに含まれる複数枚の画像に一括して処理を行います。


| Attention Branch Network:
| `Hiroshi Fukui, Tsubasa Hirakawa, Takayoshi Yamashita, Hironobu Fujiyoshi. "Attention Branch Network: Learning of Attention Mechanism for Visual Explanation". 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). <https://ieeexplore.ieee.org/document/8953929>`_

Input Information
===================

.. list-table::
   :widths: 30 70
   :class: longtable

   * - input
     -
        Attention Map Visualization の各処理を行う対象の画像ファイル一覧を含むデータセットCSVファイルを指定します。
        
        評価タブの出力結果に含まれる画像データに対しAttention Map Visualization を行うには、デフォルトのoutput_result.csvを使用します。

   * - model
     -
        Attention Map Visualizationの演算に用いるのモデルファイル（*.nnp）を指定します。
        
        評価タブで選択中の学習結果を元にAttention Map Visualization を行うには、デフォルトのresults.nnpを使用します。

   * - output
     -
        可視化結果を出力するデータセットCSVファイルのファイル名を指定します。
        
        評価タブの評価結果からAttention Map Visualization を実行した場合、学習結果フォルダに指定したファイル名で可視化結果が保存されます。

   * - attention_map_layer
     -
        Attention Branch Networkのモデルを構成する層の内、Attention Mapの層の名前を指定します。



Output Information
===================

本プラグインの実行結果は 'output' で指定した名前のCSVファイルとして出力されます。
CSVファイル内の各カラムに関しての情報は以下の通りです（以下のリストに無い名称のカラムは、 '評価' の結果得られる output_result.csvと同様の意味です）。

.. list-table::
   :widths: 30 70
   :class: longtable

   * - Attention Map Visualization
     - 対象のインスタンスへのAttention Map Visualizationの適用結果が表示されます。 元画像にjet カラーマップを重ねた形で表示され、赤色に近いほど分類結果に強く影響した箇所であることを示します。