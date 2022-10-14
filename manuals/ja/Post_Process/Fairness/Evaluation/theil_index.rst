Fairness/Theil index(Tabular)
~~~~~~~~~~~~~~~~~~~~~~
Theil indexは，データセットの個々のデータ間の利益のエントロピーを計算します．
alpha = 1 を使用すると，これは個々のデータ間の利益配分における不平等を測定します．

`Speicher, Till, Hoda Heidari, Nina Grgic-Hlaca, Krishna P. Gummadi, Adish Singla, Adrian Weller, and Muhammad Bilal Zafar. "A unified approach to quantifying algorithmic unfairness: Measuring individual &group unfairness via inequality indices." In Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining, pp. 2239-2248. 2018. <https://dl.acm.org/doi/abs/10.1145/3219819.3220046>`_

Input Information
===================

.. list-table::
   :widths: 30 70
   :class: longtable
   :header-rows: 1

   * - Property
     - Notes

   * - input
     - Theil indexが計算されるデータを含むCSVファイルを指定します．"評価"タブに表示される結果を出力するには，デフォルトの output_result.csv を使用します．

   * - target_variable
     - inputで指定したCSVファイルから，ターゲットラベルとして使用する列名を指定します．

   * - output_variable
     - inputで指定したCSVファイルから，モデル出力として使用する列名を指定します．(クラス分類の出力など．デフォルトでは"y'"が指定されます．)

   * - clf_threshold
     - 最適なクラス分類しきい値を指定します．デフォルトのしきい値は 0.5 です．

   * - fair_threshold
     - 公平性のしきい値を決定します．デフォルト値は0.10です．したがって，0.0と0.1の間のすべての結果は"公平"であり，範囲外は"不公平"であると確認することができます．

   * - num_samples
     - `Theil index`を計算するサンブル数を指定します．デフォルトでは'all'が設定されており，入力ファイルの全てのサンプルでTheil indexを計算します．

   * - output
     - `Theil index (TI)`の結果を保存するCSVファイルの名前を指定します.

Output Information
===================

このプラグインの結果はCSVファイルとして保存されます． 
CSVファイルの列の情報は下記の通りです．

.. list-table::
   :widths: 30 70
   :class: longtable

   * - Fairness Plot
     - Fairness Plot列では，データセット内の個々のデータの公平性を図として確認できます．棒グラフが緑のエリアからはみ出している場合は，個々の公平性の目標が満たされていないことを意味します．

   * - Theil index
     - Theil indexの理想的な値は0です. Theil indexが低いと公平なモデルを意味し，高い値はモデルが不公平であることを意味します．

