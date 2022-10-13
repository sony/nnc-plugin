Fairness/Equal Opportunity Difference(Tabular)
~~~~~~~~~~~~~~~~~~~~~~
Equal opportunity difference (EOD)において，モデルはグループ全体で，陽性のデータを正しく陽性と予測した割合を等しくする必要があるとしています．
バイアスが現れる可能性のある属性において多数派は'privileged'クラスと呼ばれ，バイアスが現れる可能性のある属性において少数派は'unprivileged'クラスと呼ばれます．
この指標では'unprivileged'クラスと'privileged'クラス間の感度(true positive rate)の差を計算します．

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
     - Equal opportunityが計算されるデータを含むCSVファイルを指定します．"評価"タブに表示される結果を出力するには，デフォルトの output_result.csv を使用します．

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
     - `Equal opportunity`を計算するサンブル数を指定します．デフォルトでは'all'が設定されており，入力ファイルの全てのサンプルでEqual opportunityを計算します．

   * - output
     - Equal opportunity difference (EOD) の結果を保存するCSVファイルの名前を指定します.

Output Information
===================

このプラグインの結果はCSVファイルとして保存されます． 
CSVファイルの列の情報は下記の通りです．

.. list-table::
   :widths: 30 70
   :class: longtable

   * - Fairness Plot
     - Fairness Plot列では，'privileged'属性と'unprivileged'属性間の公平性を図として確認できます．棒グラフが緑のエリアからはみ出している場合は，公平性の指標が満たされていないことを意味します．

   * - Equal Opportunity
     - Equal Opportunityの理想的な値は0です．Equal Opportunityが0を下回る場合( < 0)結果は'privileged'クラスに偏っており，Equal Opportunityが0を上回る場合( > 0)結果は'unprivileged'クラスに偏っていることを意味します．


