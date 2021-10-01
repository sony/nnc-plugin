Simple Text Classification
~~~~~~~~~~~~~~~~~~~~~~~~~~

1列目が英文文字列、2列目がカテゴリIndexであるCSVファイルを元に、Neural Network ConsoleのデータセットCSVファイル形式にコンバートします。

コンバート後のデータセットCSVファイルは、元の英文文字列が単語Index系列と単語Index系列の長さに変換されたものになります。

本プラグインが入力とするCSVファイルは、Neural Network Consoleの扱うデータセットCSVファイルとほぼ同じフォーマットであり、1行目がヘッダ、2行目以降の各行がデータを表します。CSVファイル、1列目2行目以降の各セルには英文をそのままの形で入力します。

.. code:: csv

        x:input_text,y:label
        Tomorrow's weather is sunny.,0
        This is a pen.,1
        The weather the day after tomorrow is rainy.,0
        This is an apple.,1

.. list-table::
   :widths: 30 70
   :class: longtable

   * - input
     - 変換元のCSVファイルを指定します

   * - encoding
     -
        入力CSVファイルのエンコーディングを以下から指定します
        
        ascii：ASCII
        
        shift-jis：シフトJIS
        
        utf-8：UTF-8
        
        utf-8-sig：UTF-8（BOMつき）

   * - max-length
     -
        入力文字列の最大単語長を指定します
        
        入力文字列のうち、max_lengthで指定した以降の単語は無視されます

   * - max-words
     -
        Index化する最大の単語数を指定します
        
        入力CSVファイルに含まれる単語のうち、頻出単語から順にmax-wordsで指定した数-2の単語がIndex化の対象になります
        
        その他の単語はOthersという1つの単語にまとめられます

   * - min-occurences
     -
        Index化する単語の最小出現頻度を指定します
        
        入力CSVファイルに含まれる単語のうち、min-occurencesより小さい出現回数であった単語はOthersという1つの単語にまとめられます
        
        最終的にIndex化される単語の数は、max-words、min-occurencesによって決まる数のうちの小さい方になります

   * - normalize
     - 文字コードの異なる同じ文字を統一する正規化処理を行います

   * - index-file-input
     -
        既存の単語index CSVファイルのファイル名を指定します
        
        単語index CSVファイルは、1列目が0から始まるIndex、2列目が単語で構成された、各行が単語Indexと単語からなるCSVファイルです
        
        index-file-inputを指定した場合入力CSVファイルを元にした単語Indexを用いる代わりに、ここで指定した単語index CSVファイルに記述された単語Indexを用いて入力文字列を単語Index系列に変換します

   * - index-file-output
     - 入力CSVファイルを元にして作成する単語Indexを保存する単語index CSVファイルのファイル名を指定します

   * - output-dir
     - コンバート後のデータセットCSVファイルの出力フォルダを指定します

   * - log-file-output
     - コンバート中に表示されるログファイルを保存するテキストファイルのファイル名を指定します

   * - shuffle
     -
        コンバート後のデータセットCSVファイルの各行のシャッフルを実施するかどうかを指定します
        
        true : ランダムにシャッフルを行います
        
        false : シャッフルを行いません

   * -
        output_file1
        
        output_file2
     - コンバート後のデータセットCSVファイル1,2のファイル名を指定します

   * -
        ratio1
        
        ratio2
     -
        コンバート後のデータセットCSVファイル1,2に用いるデータの割合を指定します
        
        ratio1, ratio2の割合の合計は100 (%)である必要があります
        
        ratio2が0である場合、output_file1にすべてのデータが出力されます


**ご参考**

本プラグインを用いて作成した文章分類データセットは、20newsgroups_classificationサンプルプロジェクトを用い、InputレイヤーのSizeプロパティにmax_lengthで指定した値を、EmbedレイヤーのNumClassプロパティにmax_wordsあるいはmin_occurencesにより決まる単語数を、AffineレイヤーのOutShapeプロパティに分類クラス数を指定することでひとまず学習を試行することができます。

Simple Japanese Text Classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simple Text Classificationの日本語版です。

1列目が日本語文字列、2列目がカテゴリIndexであるCSVファイルを元に、Neural Network ConsoleのデータセットCSVファイル形式にコンバートします。

コンバート後のデータセットCSVファイルは、元の日本語文字列が単語Index系列と単語Index系列の長さに変換されたものになります。

本プラグインが入力とするCSVファイルは、Neural Network Consoleの扱うデータセットCSVファイルとほぼ同じフォーマットであり、1行目がヘッダ、2行目以降の各行がデータを表します。CSVファイル、1列目2行目以降の各セルには日本語の文章をそのままの形で入力します。

.. code:: csv

        x:input_text,y:label
        明日の天気は晴れです,0
        これはペンです,1
        明後日の天気は雨です,0
        これはりんごです,1
