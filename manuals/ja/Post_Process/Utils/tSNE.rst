Utils/t-SNE
~~~~~~~~~~~

データセットCSVファイルに含まれる指定した変数のt-SNEを計算します。

Visualizing Data using t-SNE
   - \L. van der Maaten, G. Hinton.
   - http://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf

.. list-table::
   :widths: 30 70
   :class: longtable

   * - input
     - t-SNEの演算を行う変数の含まれるデータセットCSVファイルを指定します

   * - variable
     - t-SNEの演算を行う変数名を指定します

   * - dim
     - t-SNEの次元数を指定します

   * - output
     -
        inputのデータセットCSVファイルにt-SNE結果の列を追加したデータセットCSVファイルを出力するCSVファイルのファイル名を指定します
        
        評価タブの評価結果からt-SNEを実行した場合、学習結果フォルダに指定したファイル名で保存されます


**ご参考**

t-SNE結果の2次元散布図による可視化を行うには、Visualization/Scatter plotプラグインを用い、Scatter plotのinputにt-SNEの出力ファイル、xとyにt-SNE結果の各次元を指定します。

