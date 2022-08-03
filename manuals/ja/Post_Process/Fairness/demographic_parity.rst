Fairness/Demographic(statistical) Parity Difference (Tabular)
~~~~~~~~~~~~~~~~~~~~~~
Demographic Parity や Independence Parity，Statistical Parityと呼ばれる手法は，機会学習の公平性を計算するための一般的な指標です．
この指標では，保護された属性(例： 性別などのバイアスが現れるべきではない属性)において，陽性と予測したデータが同じ割合であることが公平であるとしています．
例えば，大学に入学できる確率は性別に依存しない必要があります．
簡単に言えば，結果は保護された属性から独立している必要があります．
バイアスが現れる可能性のある属性において多数派は'privileged'クラスと呼ばれ，バイアスが現れる可能性のある属性において少数派は'unprivileged'クラスと呼ばれます．
この指標は，'unprivileged'クラスと'privileged'クラスの肯定的な結果の割合の差を計算します．

`Sam Corbett-Davies, Emma Pierson, Avi Feller, Sharad Goel, and Aziz Huq. 2017.Algorithmic decision making and the cost of fairness. In Proceedings of KDD <https://dl.acm.org/doi/abs/10.1145/3097983.3098095>`_

`Cynthia Dwork, Moritz Hardt, Toniann Pitassi, Omer Reingold, and Richard Zemel. 2012. Fairness through awareness. In Proceedings of the 3rd Innovations in Theoretical Computer Science Conference (ITCS). 214–226<https://dl.acm.org/doi/abs/10.1145/2090236.2090255>`_

`Jon Kleinberg, Sendhil Mullainathan, and Manish Raghavan. 2017. Inherent Trade-Offs in the Fair Determination of Risk Scores. In Proceedings of ITCS.<https://arxiv.org/abs/1609.05807>`_

Input Information
===================

.. list-table::
   :widths: 30 70
   :class: longtable
   :header-rows: 1

   * - Property
     - Notes

   * - input
     - Demographic Parityが計算されるデータを含むCSVファイルを指定します．"評価"タブに表示される結果を出力するには，デフォルトの output_result.csv を使用します．

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
     - 	Demographic Parityを計算するサンブル数を指定します．デフォルトでは'all'が設定されており，入力ファイルの全てのサンプルでDemographic Parityを計算します．

   * - output
     - Demographic parity(DPD)の結果を保存するCSVファイルの名前を指定します.

Output Information
===================

このプラグインの結果はCSVファイルとして保存されます． 
CSVファイルの列の情報は下記の通りです．

.. list-table::
   :widths: 30 70
   :class: longtable

   * - Fairness Plot
     - Fairness Plot列では，'privileged'属性と'unprivileged'属性間の公平性を図として確認できます．棒グラフが緑のエリアからはみ出している場合は，公平性の指標が満たされていないことを意味します．

   * - Demographic Parity
     - Demographic Parityの理想的な値は0です．Demographic Parityが0を下回る場合( < 0)結果は'privileged'クラスに偏っており，Demographic Parityが0を上回る場合( > 0)結果は'unprivileged'クラスに偏っていることを意味します．