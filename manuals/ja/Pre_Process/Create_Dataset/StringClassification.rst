String Classification
~~~~~~~~~~~~~~~~~~~~~

1列目が記号列、2列目がカテゴリIndexであるCSVファイルを元に、Neural Network ConsoleのデータセットCSVファイル形式にコンバートします。Simple Text Classificationプラグインが入力文章を単語列として単語毎にIndex化するのに対し、String Classificationプラグインは文字毎にIndex化を行うため、あらゆる言語を扱う文章分類に利用することができます。

コンバート後のデータセットCSVファイルは、元の記号列が文字Index系列と文字Index系列の長さに変換されたものになります。

本プラグインが入力とするCSVファイルは、Neural Network Consoleの扱うデータセットCSVファイルとほぼ同じフォーマットであり、1行目がヘッダ、2行目以降の各行がデータを表します。CSVファイル、1列目2行目以降の各セルには文字列をそのままの形で入力します。

.. code::

   x:input_text,y:label
   (^^),0
   (;_;),1
   :-),0
   (T_T),1

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
        最大文字列長を指定します
        
        入力文字列のうち、max_lengthで指定した以降の文字は無視されます

   * - max-characters
     -
        Index化する最大の文字（記号）数を指定します
        
        入力CSVファイルに含まれる文字のうち、頻出単語から順にmax-charactersで指定した数-2の文字がIndex化の対象になります
        
        その他の文字はOthersという1つの文字にまとめられます

   * - min-occurrences
     -
        Index化する文字の最小出現頻度を指定します
        
        入力CSVファイルに含まれる文字のうち、min-occurrencesより小さい出現回数であった文字はOthersという1つの文字にまとめられます
        
        最終的にIndex化される文字の数は、max-characters、min-occurrencesによって決まる数のうちの小さい方になります

   * - normalize
     - 文字コードの異なる同じ文字を統一する正規化処理を行います

   * - index-file-input
     -
        既存の文字index CSVファイルのファイル名を指定します
        
        文字index CSVファイルは、1列目が0から始まるIndex、2列目が文字で構成された、各行が文字Indexと単語からなるCSVファイルです
        
        index-file-inputを指定した場合入力CSVファイルを元にした文字Indexを用いる代わりに、ここで指定した文字index CSVファイルに記述された文字Indexを用いて入力文字列を文字Index系列に変換します

   * - index-file-output
     - 入力CSVファイルを元にして作成する文字Indexを保存する文字index CSVファイルのファイル名を指定します

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
