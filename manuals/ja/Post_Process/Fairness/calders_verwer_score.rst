Fairness/CV Score (Tabular)
~~~~~~~~~~~~~~~~~~~~~~
CV scoreは表データに対するモデルの公平さを評価する指標です．
バイアスが現れる可能性のある属性(年齢や性別など)間で陽性と予測したデータの差を計算します．
`Fairness-aware classifier with prejudice remover regularizer. Toshihiro Kamishima, Shotaro Akaho, Hideki Asoh & Jun Sakuma.Joint European Conference on Machine Learning and Knowledge Discovery in Databases ECML PKDD 2012: Machine Learning and Knowledge Discovery in Databases pp 35–50 <https://link.springer.com/chapter/10.1007/978-3-642-33486-3_3>`_



Input Information
===================

.. list-table::
   :widths: 30 70
   :class: longtable
   :header-rows: 1

   * - Property
     - Notes

   * - input
     - CV scoreが計算されるデータを含むCSVファイルを指定します．CVスコアを計算し，"評価"タブに表示される結果を出力するには，デフォルトの output_result.csv を使用します．

   * - label_variable
     - inputで指定したCSVファイルから，正解ラベルとして使用する列名を指定します．

   * - output_variable
     - inputで指定したCSVファイルから，モデル出力として使用する列名を指定します．(クラス分類の出力など．)

   * - privileged_variable
     - inputで指定したCSVファイルから，バイアスが現れる可能性のある属性において保護されている属性として使用する列名を指定します．バイアスが現れる可能性のある属性において多数派は'privileged'クラスと呼ばれます．デフォルトでは，'female'がprivileged_variableに割り当てられます．

   * - unprivileged_variable
     - inputで指定したCSVファイルから，バイアスが現れる可能性のある属性において保護されていない属性として使用する列名を指定します．バイアスが現れる可能性のある属性において少数派は'unprivileged'クラスと呼ばれます．デフォルトでは，'male'がunprivileged_variableに割り当てられます．

   * - num_samples
     - CV scoreを計算するサンブル数を指定します．デフォルトでは'all'が設定されており，入力ファイルの全てのサンプルでCVスコアを計算します．

   * - output
     - CV scoreとaccuracyの結果を保存するCSVファイルの名前を指定します．

Output Information
===================

このプラグインの結果はCSVファイルとして保存されます．
CSVファイルの列の情報は下記の通りです．

.. list-table::
   :widths: 30 70
   :class: longtable

   * - Object Variable
     - CV score計算用に指定された正解ラベル.

   * - Output variable
     - CV score計算用に指定されたモデルの出力.

   * - Privileged variable
     - 指定されたバイアスが現れる可能性のある属性において保護されている属性．

   * - Unprivileged variable
     - 指定されたバイアスが現れる可能性のある属性において保護されていない属性．

   * - Number of samples
     - CV score計算に使用されたサンプル数．

   * - CV score
     - CV scoreの値．低いほど公平なモデルであることを意味します．

   * - Accuracy
     - モデルの精度.



