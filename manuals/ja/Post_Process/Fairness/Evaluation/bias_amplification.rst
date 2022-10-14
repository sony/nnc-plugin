Fairness/Bias Amplification
~~~~~~~~~~~~~~~~~~~~~~
Bias amplification は保護された属性間で，ターゲット属性が正解ラベルよりもどれだけ頻繁に予測されるかを測定します。．

`Wang, Angelina, and Olga Russakovsky. "Directional bias amplification." In International Conference on Machine Learning, pp. 10882-10893. PMLR, 2021. <https://arxiv.org/pdf/2102.12594.pdf>`_

Input Information
===================

.. list-table::
   :widths: 30 70
   :class: longtable
   :header-rows: 1

   * - Property
     - Notes

   * - input
     - Bias Amplification が計算されるデータを含むCSVファイルを指定します．"評価"タブに表示される結果を出力するには，デフォルトの output_result.csv を使用します．

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
     - 公平性のしきい値を決定します．デフォルト値は0.10です．したがって，-0.1と0.1の間のすべての結果は"公平"であり，範囲外は"不公平"であると確認することができます．

   * - num_samples
     - 	Bias Amplificationを計算するサンブル数を指定します．デフォルトでは'all'が設定されており，入力ファイルの全てのサンプルでBias Amplificationを計算します．

   * - output
     - Bias Amplification (BA) の結果を保存するCSVファイルの名前を指定します.

Output Information
===================

このプラグインの結果はCSVファイルとして保存されます． 
CSVファイルの列の情報は下記の通りです．

.. list-table::
   :widths: 30 70
   :class: longtable

   * - Fairness Plot
     - Fairness Plot列では，'privileged'属性と'unprivileged'属性間の公平性を図として確認できます．棒グラフが緑のエリアからはみ出している場合は，公平性の指標が満たされていないことを意味します．

   * - Bias Amplification
     - Bias Amplification が低いと公平なモデルを意味し，高い値はモデルが不公平であることを意味します．
