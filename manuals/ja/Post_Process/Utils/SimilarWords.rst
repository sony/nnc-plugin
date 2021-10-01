Utils/Similar Words
~~~~~~~~~~~~~~~~~~~

学習済みモデルに含まれる単語のEmbedパラメータを元に類似単語を求めます。



.. list-table::
   :widths: 30 70
   :class: longtable

   * - model
     -
        類似単語検索に用いる学習済みモデルファイル（*.nnp）を指定します
        
        評価タブで選択中の学習結果を元に推論を行うには、デフォルトのresults.nnpを使用します

   * - parameter
     - 学習済みモデルに含まれるパラメータのうち、類似単語検索に用いるパラメータの名前を指定します

   * - index-file-input
     -
        単語index CSVファイルのファイル名を指定します
        
        単語index CSVファイルは、1列目が0から始まるIndex、1列目が単語で構成された、各行が単語Indexと単語からなるCSVファイルです

   * - source-word
     - 類似検索を行う元となる単語を指定します

   * - num-words
     - 結果に含める類似単語数を指定します

   * - output
     - 類似単語検索結果を出力するCSVファイルのファイル名を指定します


**ご参考**

本プラグインは20newsgroups_word_embeddingサンプルプロジェクトの結果を確認するために利用することができます。

