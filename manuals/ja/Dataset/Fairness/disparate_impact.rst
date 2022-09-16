Fairness/Disparate Impact (Tabular)
~~~~~~~~~~~~~~~~~~~~~~
Disparate Impactは公平性を評価する指標です．
バイアスが現れる可能性のある属性において多数派は'privileged'クラスと呼ばれ，バイアスが現れる可能性のある属性において少数派は'unprivileged'クラスと呼ばれます．
'unprivileged'クラスと'privileged'クラスの2つのグループについて，肯定的な出力を受け取るデータの割合を比較します．
この指標は，'unprivileged'クラスの好ましい結果の率と'privileged'クラスの結果の比率として計算されます．

`Michael Feldman, Sorelle A. Friedler, John Moeller, Carlos Scheidegger, and Suresh Venkatasubramanian. "Certifying and removing disparate impact." In proceedings of the 21th ACM SIGKDD international conference on knowledge discovery and data mining, pp. 259-268. 2015. <https://arxiv.org/abs/1412.3756v3>`_

Input Information
===================

.. list-table::
   :widths: 30 70
   :class: longtable
   :header-rows: 1

   * - Property
     - Notes

   * - input
     - Disparate Impactが計算されるデータを含むCSVファイルを指定します．"評価"タブに表示される結果を出力するには，デフォルトの output_result.csv を使用します．

   * - target_variable
     - Specify the name of the column in the input CSV file to use as the target(label) variable.

   * - privileged_variable
     - inputで指定したCSVファイルから，'privileged'属性として使用する列名を指定します．(バイアスが現れる可能性のある属性において多数派は'privileged'クラスと呼ばれます．)

   * - unprivileged_variable
     - inputで指定したCSVファイルから，'unprivileged'属性として使用する列名を指定します． (バイアスが現れる可能性のある属性において少数派は'unprivileged'クラスと呼ばれます．)

   * - num_samples
     - `Disparate Impact`を計算するサンブル数を指定します．デフォルトでは'all'が設定されており，入力ファイルの全てのサンプルで`Disparate Impact`を計算します．

   * - output
     - Disparate Impact(DI)の結果を保存するCSVファイルの名前を指定します.


Output Information
===================

このプラグインの結果はCSVファイルとして保存されます． 
CSVファイルの列の情報は下記の通りです．

.. list-table::
   :widths: 30 70
   :class: longtable

   * - Fairness Plot
     - Fairness Plot列では，'privileged'属性と'unprivileged'属性間の公平性を図として確認できます．棒グラフが緑のエリアからはみ出している場合は，公平性の指標が満たされていないことを意味します．

   * - Disparate Impact
     - Disparate Impactの理想的な値は1.0です．Disparate Impactが1.0を下回る場合( < 1.0)結果は'privileged'クラスに偏っており，Disparate Impactが1.0を上回る場合( > 1.0)結果は'unprivileged'クラスに偏っていることを意味します．

