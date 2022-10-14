Fairness/Equalised Odds (Tabular)
~~~~~~~~~~~~~~~~~~~~~~
Equalized Oddsにおいて，モデルはグループ全体で陽性のデータを正しく陽性と予測した割合を等しくする必要があるとし(Equal opportunityと同じ)，それに加え陰性データを間違って陽性と予測した割合も等しくする必要があるとしています．
これは同様の結果になったデータ間でのみを公平とする制約を意味しています．
この指標は，公平性を示す指標の中でも基準が厳しいもの1つです．
バイアスが現れる可能性のある属性において多数派は'privileged'クラスと呼ばれ，バイアスが現れる可能性のある属性において少数派は'unprivileged'クラスと呼ばれます．
This metric is computed as average of absolute difference between false positive rate and true positive rate for unprivileged and privileged groups.
この指標では'unprivileged'クラスと'privileged'クラス間の感度(true positive rate)と偽陽性率(false positive rate)の絶対値の差を計算します．


`Moritz Hardt, Eric Price, and Nati Srebro. "Equality of opportunity in supervised learning." Advances in neural information processing systems 29 (2016) <https://arxiv.org/pdf/1610.02413.pdf>`_


Input Information
===================

.. list-table::
   :widths: 30 70
   :class: longtable
   :header-rows: 1

   * - Property
     - Notes

   * - input
     - Equalized Oddsが計算されるデータを含むCSVファイルを指定します．"評価"タブに表示される結果を出力するには，デフォルトの output_result.csv を使用します．

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
     - `Equalised odd`を計算するサンブル数を指定します．デフォルトでは'all'が設定されており，入力ファイルの全てのサンプルでEqualised oddを計算します．

   * - output
     - Equalised odd (AAOD) の結果を保存するCSVファイルの名前を指定します.

Output Information
===================

このプラグインの結果はCSVファイルとして保存されます． 
CSVファイルの列の情報は下記の通りです．

.. list-table::
   :widths: 30 70
   :class: longtable

   * - Fairness Plot
     - Fairness Plot列では，'privileged'属性と'unprivileged'属性間の公平性を図として確認できます．棒グラフが緑のエリアからはみ出している場合は，公平性の指標が満たされていないことを意味します．

   * - Equalised Odds
     - Equalized Oddsが低いと公平なモデルを意味し，高い値はモデルが不公平であることを意味します．




