Reject Option-Based Classification (Tabular)
~~~~~~~~~~~~~~~~~~~~~~

Reject Option-Based Classification (ROC)はモデルのバイアスを緩和する後処理手法です．
モデルの出力の不確実性が最も高い決定境界周辺で， `unprivileged`クラスには好ましい結果を仮に付与し， `privileged`クラスには好ましく無い結果を付与することで，モデルの出力の最適なしきい値とマージンを推定します．
バイアスが現れる可能性のある属性において多数派は'privileged'クラスと呼ばれ，バイアスが現れる可能性のある属性において少数派は'unprivileged'クラスと呼ばれます．

ROCの詳細については，colabチュートリアルを参照してください． : `ROC <https://colab.research.google.com/github/sony/nnabla-examples/blob/master/interactive-demos/rejection_option_based_classification.ipynb#scrollTo=k_aleVIr6GeX>`_


Citation 
===================

`Kamiran, Faisal, Asim Karim, and Xiangliang Zhang. "Decision theory for discrimination-aware classification." In 2012 IEEE 12th International Conference on Data Mining, pp. 924-929. IEEE, 2012. <https://ieeexplore.ieee.org/document/6413831>`_

Input Information
===================

.. list-table::
   :widths: 30 70
   :class: longtable
   :header-rows: 1

   * - Property
     - Notes

   * - input
     - データセットのCSVファイルを指定します．. output_result.csv は"ROC" の計算用です．

   * - target_variable
     - inputで指定したCSVファイルから，ターゲットラベルとして使用する列名を指定します．

   * - output_variable
     - inputで指定したCSVファイルから，モデル出力として使用する列名を指定します．(クラス分類の出力など．デフォルトでは"y'"が指定されます．)

   * - privileged_variable
     - inputで指定したCSVファイルから，'privileged'属性として使用する列名を指定します．(バイアスが現れる可能性のある属性において多数派は'privileged'クラスと呼ばれます．)

   * - unprivileged_variable
     - inputで指定したCSVファイルから，'unprivileged'属性として使用する列名を指定します． (バイアスが現れる可能性のある属性において少数派は'unprivileged'クラスと呼ばれます．)

   * - fair_metric
     - ROCでの最適化に利用する公平性指標を指定してください．利用できるオプションは “Demographic Parity”, “Equalised Odds”, “Equal Opportunity” です．デフォルトでは“Demographic Parity” が指定されます．

   * - metric_ub
     - 公平性指標に対する制約の上限．-1.0 から 1.0 の間で指定し，デフォルト値は 0.10 です．
   
   * - metric_lb
     - 公平性指標に対する制約の下限．-1.0 から 1.0 の間で指定し，デフォルト値は 0.10 です．

   * - output
     - Reject Option-Based Classification (ROC) の結果を保存するCSVファイルの名前を指定します. デフォルトは"roc.csv" です.

Output Information
===================

このプラグインの結果はCSVファイルとして保存されます． 
CSVファイルの列の情報は下記の通りです．


.. list-table::
   :widths: 30 70
   :class: longtable

   * - Classification Threshold
     - 指定されたfair_metricにおける推定した分類しきい値．

   * - ROC Margin
     - 指定されたfair_metricにおける推定したROCのマージン．
   
   * - Accuracy
     - ROC適用後のモデルの精度．
   
   * - Fairness metric
     - 指定された公平性指標.

Reject Option-Based Classification Predict (Tabular)
~~~~~~~~~~~~~~~~~~~~~~
このプラグインは，ROC プラグインの結果に基づいて公正なモデル予測を取得します．
注: このプラグインを実行する前に， `Reject Option-Based Classification (ROC)`プラグインを実行してください.


Input Information
===================

.. list-table::
   :widths: 30 70
   :class: longtable
   :header-rows: 1

   * - Property
     - Notes

   * - input
     - D予測ラベルの計算に使用される分類スコアを含むデータセット CSV ファイル. デフォルトはoutput_result.csvです.

   * - output_variable
     - inputで指定したCSVファイルから，モデル出力として使用する列名を指定します．(クラス分類の出力など．この場合Sigmoidなど．)

   * - privileged_variable
     - inputで指定したCSVファイルから，'privileged'属性として使用する列名を指定します．(バイアスが現れる可能性のある属性において多数派は'privileged'クラスと呼ばれます．)

   * - unprivileged_variable
     - inputで指定したCSVファイルから，'unprivileged'属性として使用する列名を指定します． (バイアスが現れる可能性のある属性において少数派は'unprivileged'クラスと呼ばれます．)

   * - roc_params
     - "Reject Option-Based Classification"プラグインで出力されたCSVファイルを指定します．デフォルトは"roc.csv" です.

   * - output
     - ROCで予測された公平な予測を出力します．. デフォルトは "roc_predict.csv"です.

Output Information
===================

このプラグインの結果はCSVファイルとして保存されます． 
CSVファイルの列の情報は下記の通りです．

.. list-table::
   :widths: 30 70
   :class: longtable

   * - ROC Predicted
     - ROCが適用された新しい分類結果．



