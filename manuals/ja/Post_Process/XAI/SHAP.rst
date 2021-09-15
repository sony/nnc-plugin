XAI/SHAP(Image)
~~~~~~~~~~~~~~~

SHAP と呼ばれる手法を用い、画像分類を行うモデルにおいて、分類結果に
影響を及ぼす入力画像の箇所を可視化します。分類において、
正の影響を及ぼした箇所が赤色、負の影響を及ぼした箇所が青色として、
元画像上に重ねて表示されます。

A Unified Approach to Interpreting Model Predictions
   - Scott Lundberg, Su-In Lee
   - https://arxiv.org/abs/1705.07874

.. list-table::
   :widths: 30 70
   :class: longtable

   * - image
     - 分析を行う画像ファイルを指定します

       評価タブの評価結果で表示されている特定の画像に対して SHAP を行うには、
       画像ファイル名の書かれたセルが選択された状態でプラグインを起動するこ
       とで、image に画像ファイル名が自動入力されます

   * - input
     -
        SHAP 処理を行う対象の画像ファイル一覧を含むデータセットCSVファイルを指定します
        
        評価タブの Output Result に含まれる画像データに対しSHAP 処理を行うには、デフォルトの output_result.csv を使用します

   * - model
     -
        SHAP 処理の演算に用いるモデルファイル（*.nnp）を指定します
        
        評価タブで選択中の学習結果を元に SHAP 処理を行うには、デフォルトのresults.nnp を使用します

   * - class_index
     -
        可視化を行うクラスの Index を指定します
        
        デフォルトでは 0 番目のクラスに対する可視化を実行します

   * - num_samples
     - 入力画像と認識結果の関係をサンプリングする回数を指定します

   * - batch_size
     - SHAP 処理の際のbatch_sizeです

   * - interim_layer
     -
        モデルを構成する層の内、input層以外の層に対してSHAP 処理を行う際に指定します
        
        input層を0番目として、何番目の層に関して SHAP 処理を行うか指定します
        
        デフォルトではinput層に対して処理が行われます

   * - output
     -
        可視化結果の画像ファイル一覧を含むデータセットCSVファイルのファイル名を指定します
        
        デフォルトでは shap.csv です

XAI/SHAP(Image batch)
~~~~~~~~~~~~~~~~~~~~~

SHAP と呼ばれる手法を用い、画像分類を行うモデルにおいて、
分類結果に影響を及ぼす入力画像の箇所を可視化します。
SHAP プラグインが 1 枚の画像に対して処理を行うのに対し、
SHAP(batch)プラグインは指定するデータセットに含まれる
複数枚の画像に一括して処理を行います。

A Unified Approach to Interpreting Model Predictions
   - Scott Lundberg, Su-In Lee
   - https://arxiv.org/abs/1705.07874

.. list-table::
   :widths: 30 70
   :class: longtable

   * - input
     -
        SHAP 処理を行う対象の画像ファイル一覧を含むデータセットCSVファイルを指定します
        
        評価タブの Output Result に含まれる画像データに対しSHAP 処理を行うには、デフォルトの output_result.csv を使用します

   * - model
     -
        SHAP 処理の演算に用いるモデルファイル（*.nnp）を指定します
        
        評価タブで選択中の学習結果を元に SHAP 処理を行うには、デフォルトのresults.nnp を使用します

   * - input_variable
     - input で指定したデータセット CSV ファイルに含まれる変数より、SHAP 処理対象の画像の変数名を指定します

   * - label_variable
     - input で指定したデータセット CSV ファイルに含まれる変数より、可視化を行うクラスの Index の変数名を指定します

   * - num_samples
     - 入力画像と認識結果の関係をサンプリングする回数を指定します

   * - batch_size
     - SHAP 処理の際のbatch_sizeです

   * - interim_layer
     -
        モデルを構成する層の内、input層以外の層に対してSHAP 処理を行う際に指定します
        
        input層を0番目として、何番目の層に関して SHAP 処理を行うか指定します
        
        デフォルトではinput層に対して処理が行われます

   * - output
     -
        可視化結果の画像ファイル一覧を含むデータセットCSVファイルのファイル名を指定します
        
        デフォルトでは shap.csv です

XAI/Kernel SHAP (Tabular)
~~~~~~~~~~~~~~~~~~~~~~~~~

Kernel SHAP と呼ばれる手法を用い、テーブルデータを用いた分類を行うモデルにおいて、
ある分類結果に関する入力データの各特徴量の寄与度を表します。

A Unified Approach to Interpreting Model Predictions
   - Scott Lundberg, Su-In Lee
   - https://arxiv.org/abs/1705.07874

.. list-table::
   :widths: 30 70
   :class: longtable
   :header-rows: 1

   * - Property
     - Notes

   * - model
     - Kernel SHAP 処理の演算に用いるモデルファイル（*.nnp）を指定します。評価タブで選択中の学習結果を元に Kernel SHAP処理を行うには、デフォルトのresults.nnp を使用します。

   * - input
     - Kernel SHAP 処理を行う対象のデータを含むデータセットCSVファイルを指定します。

   * - train
     - モデルの学習時に用いたデータ一覧を含むデータセットCSVファイルを指定します。

   * - index
     - input のCSVファイル内における、対象データの Index を指定します。

   * - alpha
     - Ridge回帰における正則化項の定数を指定します。

   * - class_index
     - 分析を行うクラスの Index を指定します。デフォルトでは 0 番目のクラスに対する分析を実行します。回帰、二値分類ではclass indexは0のみ有効です。

   * - output
     - 結果を出力するCSVファイルのファイル名を指定します。デフォルトでは shap_tabular.csv です。



XAI/Kernel SHAP (Tabular Batch)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Kernel SHAP と呼ばれる手法を用い、テーブルデータを用いた分類を行うモデルにおいて、
ある分類結果に関する入力データの各特徴量の寄与度を表します。
Kernel SHAP(tabular) プラグインが 1 レコード分のデータに対して処理を行うのに対し、
Kernel SHAP(tabular batch) プラグインは指定するデータセットCSVに含まれる
複数レコードのデータを一括して処理します。

.. list-table::
   :widths: 30 70
   :class: longtable
   :header-rows: 1

   * - Property
     - Notes

   * - model
     - Kernel SHAP 処理の演算に用いるモデルファイル（*.nnp）を指定します。評価タブで選択中の学習結果を元に Kernel SHAP処理を行うには、デフォルトのresults.nnp を使用します。

   * - input
     - Kernel SHAP 処理を行う対象のデータを含むデータセットCSVファイルを指定します。

   * - train
     - モデルの学習時に用いたデータ一覧を含むデータセットCSVファイルを指定します。

   * - class_index
     - 分析を行うクラスの Index を指定します。デフォルトでは 0 番目のクラスに対する分析を実行します。回帰、二値分類ではclass indexは0のみ有効です。

   * - alpha
     - Ridge回帰における正則化項の定数を指定します。

   * - output
     - 結果を出力するCSVファイルのファイル名を指定します。デフォルトでは shap_tabular.csv です。
