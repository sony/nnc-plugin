Utils/Simple Text Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

入力文字列のIndex系列xと、入力文字列長l（小文字のエル）を元に次の単語のIndexを予測する学習済みモデルを用いて、英語文章を生成します。

出力文章は、生成した文字列がモデルの最大文章長に達するか、モデルがEoS（End of Sentence）を示す0番目のカテゴリを出力するか、あるいは同じ単語が連続するようになるまで生成されます。



.. list-table::
   :widths: 30 70
   :class: longtable

   * - model
     -
        文章生成に用いる学習済みモデルファイル（*.nnp）を指定します
        
        評価タブで選択中の学習結果を元に推論を行うには、デフォルトのresults.nnpを使用します

   * - input-variable
     - 学習済みモデルにおいて、入力文字列のIndex系列を示す変数名を指定します

   * - length-variable
     - 学習済みモデルにおいて、入力文字列のIndex系列の長さを示す変数名を指定します

   * - index-file-input
     -
        単語index CSVファイルのファイル名を指定します
        
        単語index CSVファイルは、1列目が0から始まるIndex、2列目が単語で構成された、各行が単語Indexと単語からなるCSVファイルです

   * - seed-text
     -
        文章生成の元になるテキストを指定します
        
        ここで入力した文字列に続く文章が生成されます

   * - normalize
     - seed-textで入力した文章について文字コードの異なる同じ文字を統一する正規化処理を行います

   * - mode
     -
        文章生成の方法を指定します
        
        sampling：各単語の予測確率に基づきサンプリングを行い次の単語を決定します
        
        beam-search：beam-searchを用い、生成文章全体の生成確率が高い候補を残しながら次の単語を決定していきます

   * - temparature
     -
        samplingモード使用時の温度パラメータを指定します
        
        大きい値を指定するほど確率の低い単語も選ばれやすくなります
        
        小さい値を指定するほど最大確率の単語が選ばれやすくなります

   * - num-text
     - 生成する文章の数を指定します

   * - output
     - 文章生成結果を出力するCSVファイルのファイル名を指定します


**ご参考**

本プラグインは20newsgroups_lstm_language_model、20newsgroups_transformer_language_modelサンプルプロジェクトの結果を確認するために利用することができます。

Utils/Simple Japanese Text Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

入力文字列のIndex系列xと、入力文字列長l（小文字のエル）を元に次の単語のIndexを予測する学習済みモデルを用いて、日本語文章を生成します。

本プラグインの設定方法についてはSimple Text Generationプラグインについての解説をご参照ください。

