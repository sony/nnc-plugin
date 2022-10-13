Fairness/CV Score (Tabular)
~~~~~~~~~~~~~~~~~~~~~~
CV scoreは表データに対するモデルの公平さを評価する指標です．
バイアスが現れる可能性のある属性において多数派は'privileged'クラスと呼ばれ，バイアスが現れる可能性のある属性において少数派は'unprivileged'クラスと呼ばれます． 
'unprivileged'クラスと'privileged'クラスの2つのグループ間で陽性と予測したデータの差を計算します．
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

   * - target_variable
     - inputで指定したCSVファイルから，ターゲットラベルとして使用する列名を指定します．

   * - output_variable
     - inputで指定したCSVファイルから，モデル出力として使用する列名を指定します．(クラス分類の出力など．デフォルトでは"y'"が指定されます．)

   * - privileged_variable
     - inputで指定したCSVファイルから，'privileged'属性として使用する列名を指定します．(バイアスが現れる可能性のある属性において多数派は'privileged'クラスと呼ばれます．)

   * - unprivileged_variable
     - inputで指定したCSVファイルから，'unprivileged'属性として使用する列名を指定します． (バイアスが現れる可能性のある属性において少数派は'unprivileged'クラスと呼ばれます．)

   * - clf_threshold
     - 最適なクラス分類しきい値を指定します．デフォルトのしきい値は 0.5 です．

   * - fair_threshold
     - 公平性のしきい値を決定します．デフォルト値は0.10です．したがって，0.0と0.1の間のすべての結果は"公平"であり，範囲外は"不公平"であると確認することができます．
  
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

   * - Fairness Plot
     - Fairness Plot列では，'privileged'属性と'unprivileged'属性間の公平性を図として確認できます．棒グラフが緑のエリアからはみ出している場合は，公平性の指標が満たされていないことを意味します．
   
   * - CV score
     - CV scoreの値．低いほど公平なモデルであることを意味します．

   * - Accuracy
     - モデルの精度.



