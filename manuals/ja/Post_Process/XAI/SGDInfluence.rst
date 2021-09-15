XAI/SGD Influence
~~~~~~~~~~~~~~~~~

SGD Influence と呼ばれる手法を用い、画像認識を行うモデルにおいて入力画像が学習結果（精度）へ与える影響をスコアとして算出します。学習に悪影響を及ぼした順に並び替えてデータセットとスコアを表示します。

本プラグインでは、以下の3論文をベースとした近似アルゴリズムを使用しています。

Data Cleansing for Deep Neural Networks with Storage-efficient Approximation of Influence Functions
  - Kenji Suzuki, Yoshiyuki Kobayashi, Takuya Narihira
  - https://arxiv.org/abs/2103.11807
Data Cleansing for Models Trained with SGD
  - Satoshi Hara, Atsushi Nitanda, Takanori Maehara
  - https://arxiv.org/abs/1906.08473
Understanding Black-box Predictions via Influence Functions
  - Pang Wei Koh, Percy Liang
  - https://arxiv.org/abs/1703.04730

.. list-table::
   :widths: 30 70
   :class: longtable

   * - input-train
     -
        SGD Influence処理を行う対象の画像ファイル一覧を含むデータセットCSVファイルを指定します
        
        こちらのデータセットに含まれる各画像についてスコアが算出されます

   * - input-val
     -
        画像ファイル一覧を含むCSVファイルを指定します
        
        SGD Influence処理を行う際には、評価を行う対象であるinput-train以外に内部計算用のデータセットが必要となります
        
        そのために、input-trainに指定したCSVファイルとは異なるデータセットCSVファイルを指定します

   * - output
     - 評価結果を出力するCSVファイルのファイル名を指定します

   * - seed
     -
        乱数生成のためのseedを指定します
        
        input-train で指定されたデータセットのシャッフルに用いられます

   * - model
     -
        SGD Influenceの演算に用いるモデルファイル（*.nnp）を指定します
        
        評価タブで選択中の学習結果を元に計算を行うには、デフォルトのresults.nnp を使用します

   * - batch_size
     - SGD Influenceのモデルが学習する際のbatch_sizeです

XAI/SGD Influence (tabular)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

SGD Influence と呼ばれる手法を用い、テーブルデータを用いた分類を行うモデルにおいて、ある分類結果に関する入力データが学習結果へ与える影響をスコアとして算出します。学習に悪影響を及ぼした順に並び替えてデータセットとスコアを表示します。

**ご注意** *現在、dropoutが含まれるネットワークにおいて、当プラグインが動作しない不具合が確認されています。*


Data Cleansing for Models Trained with SGD
  - Satoshi Hara, Atsushi Nitanda, Takanori Maehara
  - https://arxiv.org/abs/1906.08473

.. list-table::
   :widths: 30 70
   :class: longtable

   * - model
     - SGD Influenceの演算に用いるモデルファイル（*.nnp）を指定します
       評価タブで選択中の学習結果を元に計算を行うには、デフォルトのresults.nnp を使用します

   * - batch_size
     - SGD Influenceのモデルが学習する際のbatch_sizeです

   * - input-train
     - SGD Influence処理を行う対象のデータを含むデータセットCSVファイルを指定します。
       こちらのデータセットに対してスコアが算出されます。

   * - input-val
     - input-trainに指定したCSVファイルとは異なるデータセットCSVファイルを指定します。
       SGD Influence処理を行う際には、評価を行う対象であるinput-train以外に
       内部計算用のデータセットが必要となります。

   * - output
     - 評価結果を出力するCSVファイルのファイル名を指定します

   * - seed
     - 乱数生成のためのseedを指定します
       input-train で指定されたデータセットのシャッフルに用いられます


