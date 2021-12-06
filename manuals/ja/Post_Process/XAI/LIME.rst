XAI/LIME(image)
~~~~~~~~~~~~~~~

LIMEと呼ばれる手法を用い、画像分類を行うモデルにおいて、分類結果に影響を及ぼす入力画像の箇所を可視化します。

`Marco Tulio Ribeiro, Sameer Singh, Carlos Guestrin. "Why Should I Trust You?: Explaining the Predictions of Any Classifier". Knowledge Discovery and Data Mining, 2016. <https://dl.acm.org/doi/abs/10.1145/2939672.2939778>`_

Input Information
===================

.. list-table::
   :widths: 30 70
   :class: longtable

   * - model
     -
        LIMEの演算に用いるモデルファイル（*.nnp）を指定します
        
        評価タブで選択中の学習結果を元にLIMEを行うには、デフォルトのresults.nnpを使用します

   * - image
     -
        分析を行う画像ファイルを指定します。
        
        評価タブの評価結果で表示されている特定の画像に対してLIMEを行うには、画像ファイル名の係れたセルが選択された状態でプラグインを起動することで、imageに画像ファイル名が自動入力されます

   * - class_index
     -
        可視化を行うクラスのIndexを指定します
        
        デフォルトでは0番目のクラスに対する可視化を実行します

   * - num_samples
     - 入力画像と認識結果の関係をサンプリングする回数を指定します

   * - num_segments
     - 入力画像を分割するセグメントの数を指定します

   * - num_segments_2
     - num_segmentsに分割された領域のうち、可視化するセグメントの数を指定します

   * - output
     -
        可視化結果を出力する画像ファイルのファイル名を指定します
        
        評価タブの評価結果からLIMEを実行した場合、学習結果フォルダに指定したファイル名で可視化結果が保存されます

Output Information
===================

本プラグインの実行結果は 'output' で指定した名前のPNGファイルとして出力されます。
対象のインスタンスへの本プラグインの適用結果が表示されます。 


XAI/LIME(image batch)
~~~~~~~~~~~~~~~~~~~~~

LIMEと呼ばれる手法を用い、画像分類を行うモデルにおいて、分類結果に影響を及ぼす入力画像の箇所を可視化します。LIME プラグインが 1 枚の画像に対して処理を行うのに対し、LIME(batch)プラグインは指定するデータセットに含まれる複数枚の画像に一括して処理を行います。二値分類もしくは多値分類に用いることができます。

`Marco Tulio Ribeiro, Sameer Singh, Carlos Guestrin. "Why Should I Trust You?: Explaining the Predictions of Any Classifier". Knowledge Discovery and Data Mining, 2016. <https://dl.acm.org/doi/abs/10.1145/2939672.2939778>`_

Input Information
===================

.. list-table::
   :widths: 30 70
   :class: longtable

   * - input
     -
        LIME 処理を行う対象の画像ファイル一覧を含むデータセットCSVファイルを指定します
        
        評価タブの出力結果に含まれる画像データに対しLIME 処理を行うには、デフォルトの output_result.csv を使用します

   * - model
     -
        LIMEの演算に用いるモデルファイル（*.nnp）を指定します
        
        評価タブで選択中の学習結果を元にLIMEを行うには、デフォルトのresults.nnpを使用します

   * - input_variable
     - input で指定したデータセットCSVファイルに含まれる変数より、LIME 処理対象の画像の変数名を指定します

   * - label_variable
     - input で指定したデータセットCSVファイルに含まれる変数より、可視化を行うクラスの Index の変数名を指定します

   * - num_samples
     - 入力画像と認識結果の関係をサンプリングする回数を指定します

   * - num_segments
     - 入力画像を分割するセグメントの数を指定します

   * - num_segments_2
     - num_segmentsに分割された領域のうち、可視化するセグメントの数を指定します

   * - output
     -
        可視化結果の画像ファイル一覧を含むデータセットCSVファイルのファイル名を指定します
        
        デフォルトでは lime.csv です

Output Information
===================

本プラグインの実行結果は 'output' で指定した名前のCSVファイルとして出力されます。
CSVファイル内の各カラムに関しての情報は以下の通りです（以下のリストに無い名称のカラムは、 '評価' の結果得られる output_result.csvと同様の意味です）。

.. list-table::
   :widths: 30 70
   :class: longtable

   * - lime
     - 対象のインスタンスへの本プラグインの適用結果が表示されます

XAI/LIME(tabular)
~~~~~~~~~~~~~~~~~

LIME と呼ばれる手法を用い、テーブルデータを用いた分類を行うモデルにおいて、ある分類結果に関する入力データの各特徴量の寄与を、各特徴量の不等式とその寄与度として表します。回帰モデルや、特徴量にカテゴリ変数を含む分類モデルにも対応しています。

`Marco Tulio Ribeiro, Sameer Singh, Carlos Guestrin. "Why Should I Trust You?: Explaining the Predictions of Any Classifier". Knowledge Discovery and Data Mining, 2016. <https://dl.acm.org/doi/abs/10.1145/2939672.2939778>`_

Input Information
===================

.. list-table::
   :widths: 30 70
   :class: longtable

   * - model
     -
        LIMEの演算に用いるモデルファイル（*.nnp）を指定します
        
        評価タブで選択中の学習結果を元にLIMEを行うには、デフォルトのresults.nnpを使用します

   * - input
     - LIME 処理を行う対象のデータを含むデータセットCSVファイルを指定します

   * - categorical
     - input のCSVファイル内における、カテゴリ変数が入力されている列番号をカンマ区切りの整数で指定します

   * - index
     - input のCSVファイル内における、対象データの Index を指定します。

   * - class_index
     -
        分析を行うクラスの Index を指定します
        
        デフォルトでは 0 番目のクラスに対する分析を実行します
        
        回帰、二値分類ではclass indexは0のみ有効です

   * - num_samples
     - 入力データと分類結果の関係をサンプリングする回数を指定します

   * - train
     - モデルの学習時に用いたデータ一覧を含むデータセットCSVファイルを指定します

   * - output
     -
        結果を出力するCSVファイルのファイル名を指定します
        
        デフォルトでは lime_tabular.csv です。

Output Information
===================

本プラグインの実行結果は 'output' で指定した名前のCSVファイルとして出力されます。
CSVファイル内の各行と各カラムに関しての情報は以下の通りです。
'Sample (Index {n})' の行は各特徴量の値に対応します（ '評価' の結果得られる output_result.csvと同様の意味です）。
'Importance' の行は、分類結果における各特徴量の寄与度を表します。'Importance' の行の上の行は、その寄与度を与える際の各特徴量の不等式を示します。


XAI/LIME(tabular batch)
~~~~~~~~~~~~~~~~~~~~~~~

LIME と呼ばれる手法を用い、テーブルデータを用いた分類を行うモデルにおいて、ある分類結果に関する入力データの各特徴量の寄与を、各特徴量の不等式とその寄与度として表します。回帰モデルや、特徴量にカテゴリ変数を含む分類モデルにも対応しています。LIME(tabular) プラグインが 1 レコード分のデータに対して処理を行うのに対し、LIME(tabular batch) プラグインは指定するデータセットCSVに含まれる複数レコードのデータを一括して処理します。

`Marco Tulio Ribeiro, Sameer Singh, Carlos Guestrin. "Why Should I Trust You?: Explaining the Predictions of Any Classifier". Knowledge Discovery and Data Mining, 2016. <https://dl.acm.org/doi/abs/10.1145/2939672.2939778>`_

Input Information
===================

.. list-table::
   :widths: 30 70
   :class: longtable

   * - model
     -
        LIMEの演算に用いるモデルファイル（*.nnp）を指定します
        
        評価タブで選択中の学習結果を元にLIMEを行うには、デフォルトのresults.nnpを使用します

   * - input
     - LIME 処理を行う対象のデータを含むデータセットCSVファイルを指定します

   * - categorical
     - input のCSVファイル内における、カテゴリ変数が入力されている列番号をカンマ区切りの整数で指定します

   * - class_index
     -
        分析を行うクラスの Index を指定します
        
        デフォルトでは 0 番目のクラスに対する分析を実行します
        
        回帰、二値分類ではclass indexは0のみ有効です

   * - num_samples
     - 入力データと分類結果の関係をサンプリングする回数を指定します

   * - train
     - モデルの学習時に用いたデータ一覧を含むデータセットCSVファイルを指定します

   * - output
     -
        結果を出力するCSVファイルのファイル名を指定します
        
        デフォルトでは lime_tabular.csv です。

Output Information
===================

本プラグインの実行結果は 'output' で指定した名前のCSVファイルとして出力されます。
CSVファイル内の各行と各カラムに関しての情報は以下の通りです。
以下のリストに無い名称のカラムは、 対象のインスタンスに対する各特徴量の寄与度を表します。

.. list-table::
   :widths: 30 70
   :class: longtable

   * - index
     - 対象のインスタンスの `input-train`のデータセットCSVファイルにおけるインデックスを意味します
