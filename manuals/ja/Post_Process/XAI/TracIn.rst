XAI/TracIn
~~~~~~~~~~

TracIn と呼ばれる手法を用い、画像認識を行うモデルにおいて入力画像が学習結果（精度）へ与える影響をスコアとして算出します。学習に悪影響を及ぼした順に並び替えてデータセットとスコアを表示します。

本プラグインはGPU利用時のみ動作します。

`Garima Pruthi, Frederick Liu, Satyen Kale, Mukund Sundararajan. "Estimating Training Data Influence by Tracing Gradient Descent". In Advances in Neural Information Processing Systems, 2020. <https://papers.nips.cc/paper/2020/file/e6385d39ec9394f2f3a354d9d2b88eec-Paper.pdf>`_


Input Information
===================

.. list-table::
   :widths: 30 70
   :class: longtable

   * - input-train
     -
        TracIn 処理を行う対象の画像ファイル一覧を含むデータセットCSVファイルを指定します
        
        こちらのデータセットに対してスコアが算出されます

   * - model
     -
        TracIn処理を行う際に用いるモデルを指定します
        
        resnet23もしくはresnet56から選びます

   * - output
     - 評価結果を出力するCSVファイルのファイル名を指定します

   * - train_batch_size
     - TracInのモデルが学習する際のbatch_sizeです

   * - train_epochs
     - TracInのモデルが学習する際のepoch数です

   * - seed
     -
        乱数生成のためのseedを指定します
        
        data augmentationの際に用いられます
        

Output Information
===================

本プラグインの実行結果は 'output' で指定した名前のCSVファイルとして出力されます。
CSVファイル内の各カラムに関しての情報は以下の通りです（以下のリストに無い名称のカラムは、 '評価' の結果得られる output_result.csvと同様の意味です）。

.. list-table::
   :widths: 30 70
   :class: longtable

   * - influence
     - TracInによって計算された、対象のインスタンスのInfuence値です。本CSVファイルの行はこのInfluence値によってソートされています

   * - datasource_index
     - 対象のインスタンスの `input-train`のデータセットCSVファイルにおけるインデックスを意味します。再学習の際などに、 `input-train` のデータセットCSVファイルと同じ順番に並べ替える際に利用します

Link
========
| アルゴリズムに関する詳細な説明に関してましては以下のリンクをご参照ください。
| https://github.com/sony/nnabla-examples/tree/master/responsible_ai/tracin#overview


