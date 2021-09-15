XAI/SmoothGrad
~~~~~~~~~~~~~~

SmoothGrad と呼ばれる手法を用い、画像分類を行うモデルにおいて、分類結果に影響を及ぼす入力画像の箇所を可視化します。

SmoothGrad: removing noise by adding noise
   - Daniel Smilkov, Nikhil Thorat, Been Kim, Fernanda Viégas, Martin Wattenberg
   - https://arxiv.org/abs/1706.03825

.. list-table::
   :widths: 30 70
   :class: longtable

   * - model
     -
        SmoothGrad の演算に用いるモデルファイル（*.nnp）を指定します
        
        評価タブで選択中の学習結果を元に SmoothGrad を行うには、デフォルトのresults.nnp を使用します

   * - noise_level
     -
        標準偏差を計算するためのノイズレベル(0.0 - 1.0)を指定します
        
        この標準偏差は SmoothGrad の演算に用いるガウシアンノイズを生成するために用いられます

   * - num_samples
     -
        勾配計算を行う回数を指定します
        
        各回においてガウシアンノイズを加えた入力画像を用いて勾配計算を行い、平均をとることで最終的な感度マップを得ます

   * - image
     -
        分析を行う画像ファイルを指定します
        
        評価タブの評価結果で表示されている特定の画像に対して SmoothGrad を行うには、画像ファイル名の係れたセルが選択された状態でプラグインを起動することで、image に画像ファイル名が自動入力されます

   * - class_index
     - 可視化を行うクラスの Index を指定します。デフォルトでは 0 番目のクラスに対する可視化を実行します

   * - output
     -
        可視化結果を出力する画像ファイルのファイル名を指定します
        
        評価タブの評価結果から SmoothGrad を実行した場合、学習結果フォルダに指定したファイル名で可視化結果が保存されます

XAI/SmoothGrad (batch)
~~~~~~~~~~~~~~~~~~~~~~

SmoothGrad と呼ばれる手法を用い、画像分類を行うモデルにおいて、分類結果に影響を及ぼす入力画像の箇所を可視化します。SmoothGrad プラグインが 1 枚の画像に対して処理を行うのに対し、SmoothGrad(batch)プラグインは指定するデータセットに含まれる複数枚の画像に一括して処理を行います。

SmoothGrad: removing noise by adding noise
   - Daniel Smilkov, Nikhil Thorat, Been Kim, Fernanda Viégas, Martin Wattenberg
   - https://arxiv.org/abs/1706.03825

.. list-table::
   :widths: 30 70
   :class: longtable

   * - model
     -
        SmoothGrad の演算に用いるモデルファイル（*.nnp）を指定します
        
        評価タブで選択中の学習結果を元に SmoothGrad を行うには、デフォルトのresults.nnp を使用します

   * - noise_level
     -
        標準偏差を計算するためのノイズレベル(0.0 - 1.0)を指定します
        
        この標準偏差は SmoothGrad の演算に用いるガウシアンノイズを生成するために用いられます

   * - num_samples
     -
        勾配計算を行う回数を指定します
        
        各回においてガウシアンノイズを加えた入力画像を用いて勾配計算を行い、平均をとることで最終的な感度マップを得ます

   * - input
     -
        SmoothGrad 処理を行う対象の画像ファイル一覧を含むデータセットCSVファイルを指定します
        
        評価タブの Output Result に含まれる画像データに対し SmoothGrad 処理を行うには、デフォルトの output_result.csv を使用します

   * - output
     -
        可視化結果の画像ファイル一覧を含むデータセットCSVファイルのファイル名を指定します
        
        デフォルトでは smoothgrad.csv です

   * - class_index
     - 可視化を行うクラスの Index を指定します。デフォルトでは 0 番目のクラスに対する可視化を実行します

   * - label_variable
     - input で指定したデータセット CSV ファイルに含まれる変数より、可視化を行うクラスの Index の変数名を指定します
