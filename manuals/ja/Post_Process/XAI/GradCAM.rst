XAI/Grad-CAM
~~~~~~~~~~~~

Grad-CAMと呼ばれる手法を用い、画像分類を行うConvolutional Neural Networksにおいて、分類結果に影響を及ぼす入力画像の箇所を可視化します。

Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization

Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam,
  - Devi Parikh, Dhruv Batra
  - https://arxiv.org/abs/1610.02391

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

XAI/Grad-CAM(batch)
~~~~~~~~~~~~~~~~~~~

Grad-CAMと呼ばれる手法を用い、画像分類を行うConvolutional Neural Networksにおいて、分類結果に影響を及ぼす入力画像の箇所を可視化します。Grad-CAMプラグインが1枚の画像に対して処理を行うのに対し、Grad-CAM(batch)プラグインは指定するデータセットに含まれる複数枚の画像に一括して処理を行います。

Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization

Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam,
  - Devi Parikh, Dhruv Batra
  - https://arxiv.org/abs/1610.02391

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

