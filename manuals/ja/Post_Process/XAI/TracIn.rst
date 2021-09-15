XAI/TracIn
~~~~~~~~~~

TracIn と呼ばれる手法を用い、画像認識を行うモデルにおいて入力画像が学習結果（精度）へ与える影響をスコアとして算出します。学習に悪影響を及ぼした順に並び替えてデータセットとスコアを表示します。

本プラグインはGPU利用時のみ動作します。

Estimating Training Data Influence by Tracing Gradient Descent
   - Garima Pruthi, Frederick Liu, Mukund Sundararajan, Satyen Kale
   - https://arxiv.org/abs/2002.08484



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

