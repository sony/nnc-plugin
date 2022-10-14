Fairness/KL Divergence
~~~~~~~~~~~~~~~~~~~~~~
KL Divergenve は `しきい値不変` な公平性評価使用とも呼ばれ，しきい値に関係なく異なるグループ間で公平性を計算します．

`Chen, Mingliang, and Min Wu. "Towards threshold invariant fair classification." In Conference on Uncertainty in Artificial Intelligence, pp. 560-569. PMLR, 2020. <https://arxiv.org/pdf/2006.10667.pdf>`_

Input Information
===================

.. list-table::
   :widths: 30 70
   :class: longtable
   :header-rows: 1

   * - Property
     - Notes

   * - input
     - KL Divergence が計算されるデータを含むCSVファイルを指定します．"評価"タブに表示される結果を出力するには，デフォルトの output_result.csv を使用します．

   * - target_variable
     - inputで指定したCSVファイルから，ターゲットラベルとして使用する列名を指定します．

   * - output_variable
     - inputで指定したCSVファイルから，モデル出力として使用する列名を指定します．(クラス分類の出力など．デフォルトでは"y'"が指定されます．)

   * - privileged_variable
     - inputで指定したCSVファイルから，'privileged'属性として使用する列名を指定します．(バイアスが現れる可能性のある属性において多数派は'privileged'クラスと呼ばれます．)

   * - unprivileged_variable
     - inputで指定したCSVファイルから，'unprivileged'属性として使用する列名を指定します． (バイアスが現れる可能性のある属性において少数派は'unprivileged'クラスと呼ばれます．)

   * - fair_threshold
     - 公平性のしきい値を決定します．デフォルト値は0.10です．したがって，-0.1と0.1の間のすべての結果は"公平"であり，範囲外は"不公平"であると確認することができます．

   * - num_samples
     - KL Divergenceを計算するサンブル数を指定します．デフォルトでは'all'が設定されており，入力ファイルの全てのサンプルでKL Divergenceを計算します．

   * - output
     - KL Divergence (KL) の結果を保存するCSVファイルの名前を指定します.

Output Information
===================

The result of this plugin is saved in the designated 'output' path as CSV file.
The information on the columns of CSV file is as follows.

.. list-table::
   :widths: 30 70
   :class: longtable

   * - Fairness Plot
     - Fairness Plot列では，'privileged'属性と'unprivileged'属性間の公平性を図として確認できます．棒グラフが緑のエリアからはみ出している場合は，公平性の指標が満たされていないことを意味します．

   * - KL Divergence
     - KL Divergence が低いと公平なモデルを意味し，高い値は公平性が欠如している可能性を示します．
