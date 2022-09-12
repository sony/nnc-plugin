Fairness/Reweighing
~~~~~~~~~~~~~~~~~~~~~~~~

インスタンスに応じて重み付けをします

`Kamiran, Faisal and Calders, Toon. "Data preprocessing techniques for classification without discrimination". Knowledge and Information Systems, 33(1):1-33, 2012. <https://link.springer.com/article/10.1007/s10115-011-0463-8>`_

Input Information
===================

.. list-table::
   :widths: 30 70
   :class: longtable
   :header-rows: 1

   * - input-train
     - trainingに用いるためのデータセットCSVファイルを指定します

   * - label-name
     - input で指定したデータセットCSVファイルに含まれる変数より、評価を行うターゲットの変数名を指定します

   * - protected-attribute
     - input で指定したデータセットCSVファイルに含まれる変数より、結果にバイアスをもたらすと考えられる変数名を指定します

   * - output
     - 評価結果を出力するCSVファイルのファイル名を指定します

Output Information
===================

本プラグインの実行結果は 'output' で指定した名前のCSVファイルとして出力されます。
CSVファイル内の各カラムに関しての情報は以下の通りです。

.. list-table::
   :widths: 30 70
   :class: longtable

   * - Sample_weight
     - protected-attributeについての各サンプルの重みです

Example Use
===================

nnc-pluginのリポジトリから '_Fairness' フォルダーをダウンロードし、 'neural_network_console\libs\plugins\_Post_Process' の配下に配置します。
Neural Network Consoleの 'プロジェクトを開く' から 'neural_network_console\libs\plugins\_Post_Process\_Fairness\bias_mitigation_utils' の配下にある 'german_credit.sdcproj' を選択します。
ホーム画面のプロジェクト上にある、'german_credit.sdcproj' 上で右クリックして、'その他のツール'、 'コマンドプロンプト'を選択し、コマンドプロンプト上で 'python create_german_credit_csv.py' を入力して実行します。
Neural Network Consoleを再起動します。
Neural Network Consoleのサンプルプロジェクトである german_credit.sdcproj を用いて本プラグインを試してみることができます。
まずは学習と評価をすることで出力結果を得ます。出力結果画面上で右クリックし、'プラグイン'、 'Fairness'、'bias mitigation'の順に選択して条件入力画面を表示させます。
'input-train'にはgerman_creditのサンプルプロジェクトで用いるgerman credit datasetが自動で指定されます。
次に、'label-name' と 'protected-attribute' には input-train のCSVファイル内のカラム名を指定してみます。
'label-name' として 'y__0:Good / bad' のカラムを指定し、'protected-attribute'として'x__32:Personal status and sex=A91' を入力します。
この 'protected attribute' には性別に関しての情報が含まれており (sex=A91)、この特徴量が 信用リスク評価に影響（'y__0:Good / bad' ）するとしたら望ましくありません。
本プラグインを用いることで、'output' で指定したCSVファイルに、各サンプルの重みの算出結果を見ることができます。