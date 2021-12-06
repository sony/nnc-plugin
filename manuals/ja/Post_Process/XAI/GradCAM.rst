XAI/Grad-CAM
~~~~~~~~~~~~

Grad-CAMと呼ばれる手法を用い、画像分類を行うConvolutional Neural Networksにおいて、分類結果に影響を及ぼす入力画像の箇所を可視化します。
本pluginを実行するためには、少なくとも1つのConvolution層がモデル内に含まれている必要があります。

`Ramprasaath R Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, Dhruv Batra. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization". Proceedings of the IEEE International Conference on Computer Vision, 2017. <https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html>`_

Input Information
===================

.. list-table::
   :widths: 30 70
   :class: longtable

   * - model
     -
        Grad-CAMの演算に用いるConvolutional Neural Networksのモデルファイル（*.nnp）を指定します
        
        評価タブで選択中の学習結果を元にGrad-CAMを行うには、デフォルトのresults.nnpを使用します

   * - image
     -
        分析を行う画像ファイルを指定します。
        
        評価タブの評価結果で表示されている特定の画像に対してGrad-CAMを行うには、画像ファイル名の係れたセルが選択された状態でプラグインを起動することで、imageに画像ファイル名が自動入力されます

   * - class_index
     -
        可視化を行うクラスのIndexを指定します
        
        デフォルトでは0番目のクラスに対する可視化を実行します

   * - output
     -
        可視化結果を出力する画像ファイルのファイル名を指定します
        
        評価タブの評価結果からGrad-CAMを実行した場合、学習結果フォルダに指定したファイル名で可視化結果が保存されます


Output Information
===================

本プラグインの実行結果は 'output' で指定した名前のPNGファイルとして出力されます。
対象のインスタンスへの本プラグインの適用結果が表示されます。 元画像にjet カラーマップを重ねた形で表示され、赤色に近いほど分類結果に強く影響した箇所であることを示します。


XAI/Grad-CAM(batch)
~~~~~~~~~~~~~~~~~~~

Grad-CAMと呼ばれる手法を用い、画像分類を行うConvolutional Neural Networksにおいて、分類結果に影響を及ぼす入力画像の箇所を可視化します。Grad-CAMプラグインが1枚の画像に対して処理を行うのに対し、Grad-CAM(batch)プラグインは指定するデータセットに含まれる複数枚の画像に一括して処理を行います。
本pluginを実行するためには、少なくとも1つのConvolution層がモデル内に含まれている必要があります。

`Ramprasaath R Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, Dhruv Batra. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization". Proceedings of the IEEE International Conference on Computer Vision, 2017. <https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html>`_

Input Information
===================

.. list-table::
   :widths: 30 70
   :class: longtable

   * - input
     -
        Grad-CAM処理を行う対象の画像ファイル一覧を含むデータセットCSVファイルを指定します
        
        評価タブの出力結果に含まれる画像データに対しGrad-CAM処理を行うには、デフォルトのoutput_result.csvを使用します

   * - model
     -
        Grad-CAMの演算に用いるConvolutional Neural Networksのモデルファイル（*.nnp）を指定します
        
        評価タブで選択中の学習結果を元にGrad-CAMを行うには、デフォルトのresults.nnpを使用します

   * - input_variable
     - inputで指定したデータセットCSVファイルに含まれる変数より、Grad-CAM処理対象の画像の変数名を指定します

   * - label_variable
     - inputで指定したデータセットCSVファイルに含まれる変数より、可視化を行うクラスのIndexの変数名を指定します

   * - output
     -
        可視化結果を出力するデータセットCSVファイルのファイル名を指定します
        
        評価タブの評価結果からGrad-CAMを実行した場合、学習結果フォルダに指定したファイル名で可視化結果が保存されます

Output Information
===================

本プラグインの実行結果は 'output' で指定した名前のCSVファイルとして出力されます。
CSVファイル内の各カラムに関しての情報は以下の通りです（以下のリストに無い名称のカラムは、 '評価' の結果得られる output_result.csvと同様の意味です）。

.. list-table::
   :widths: 30 70
   :class: longtable

   * - gradcam
     - 対象のインスタンスへの本プラグインの適用結果が表示されます。 元画像にjet カラーマップを重ねた形で表示され、赤色に近いほど分類結果に強く影響した箇所であることを示します
