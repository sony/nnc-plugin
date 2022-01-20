XAI/XAI Visualization
~~~~~~~~~~~~~~~~~~~~~

画像分類を行うモデルにおいて、分類結果に影響を及ぼす入力画像の箇所をGrad-CAM、LIME、SHAP、SmoothGradを用いて一度に可視化します。

| Grad-CAM:
| `Ramprasaath R Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, Dhruv Batra. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization". Proceedings of the IEEE International Conference on Computer Vision, 2017. <https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html>`_
|
| LIME:
| `Marco Tulio Ribeiro, Sameer Singh, Carlos Guestrin. "Why Should I Trust You?: Explaining the Predictions of Any Classifier". Knowledge Discovery and Data Mining, 2016. <https://dl.acm.org/doi/abs/10.1145/2939672.2939778>`_
|
| SHAP:
| `Scott M Lundberg, Su-In Lee. "A unified approach to interpreting model predictions". Proceedings of the 31st International Conference on Neural Information Processing Systems, 2017. <https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html>`_
|
| SmoothGrad:
| `Daniel Smilkov, Nikhil Thorat, Been Kim, Fernanda Viégas, Martin Wattenberg. "Smoothgrad: removing noise by adding noise". Workshop on Visualization for Deep Learning, ICML, 2017. <https://arxiv.org/abs/1706.03825>`_

Input Information
===================

.. list-table::
   :widths: 30 70
   :class: longtable

   * - model
     -
        XAI Visualization の各演算に用いるのモデルファイル（*.nnp）を指定します。
        
        評価タブで選択中の学習結果を元にXAI Visualization を行うには、デフォルトのresults.nnpを使用します。

   * - image
     -
        分析を行う画像ファイルを指定します。
        
        評価タブの評価結果で表示されている特定の画像に対してXAI Visualization を行うには、画像ファイル名の係れたセルが選択された状態でプラグインを起動することで、imageに画像ファイル名が自動入力されます。

   * - class_index
     -
        可視化を行うクラスのIndexを指定します。
        
        デフォルトでは0番目のクラスに対する可視化を実行します。

   * - output
     -
        4つの可視化結果を出力するデータセットCSVファイルのファイル名を指定します。
        
        評価タブの評価結果からXAI Visualization を実行した場合、学習結果フォルダに指定したファイル名で可視化結果が保存されます。
    
   * - num_segments 
     - LIME 処理の際に、可力画像を分割するセグメントの数を指定します。
    
   * - num_segments_2 
     - LIME 処理の際に、num_segmentsに分割された領域のうち、可視化するセグメントの数を指定しま す。
    
   * - num_samples_lime
     - LIME 処理の際に、入力画像と認識結果の関係をサンプリングする回数を指定します。
　　
   * - num_samples_shap
     - SHAP 処理の際に、入力画像と認識結果の関係をサンプリングする回数を指定します。

   * - num_samples_smoothgrad
     - SmoothGrad 処理の際に、入力画像と認識結果の関係をサンプリングする回数を指定します。
    
   * - input
     -
        SHAP 処理を行う対象の画像ファイル一覧を含むデータセットCSVファイルを指定します。
        
        評価タブの Output Result に含まれる画像データに対しSHAP 処理を行うには、デフォルトの output_result.csv を使用します。

   * - batch_size
     - SHAP 処理の際のbatch_sizeです。

   * - interim_layer
     -
        モデルを構成する層の内、input層以外の層に対してSHAP 処理を行う際に指定します。
        
        input層を0番目として、何番目の層に関して SHAP 処理を行うか指定します。
        
        デフォルトではinput層に対して処理が行われます。

   * - noise_level
     -
        標準偏差を計算するためのノイズレベル(0.0 - 1.0)を指定します。
        
        この標準偏差は SmoothGrad の演算に用いるガウシアンノイズを生成するために用いられます。

Output Information
===================

本プラグインの実行結果は 'output' で指定した名前のCSVファイルとして出力されます。
CSVファイル内の各カラムに関しての情報は以下の通りです（以下のリストに無い名称のカラムは、 '評価' の結果得られる output_result.csvと同様の意味です）。

.. list-table::
   :widths: 30 70
   :class: longtable

   * - Grad-CAM
     - 対象のインスタンスへのGrad-CAM の適用結果が表示されます。 元画像にjet カラーマップを重ねた形で表示され、赤色に近いほど分類結果に強く影響した箇所であることを示します。

   * - LIME
     - 対象のインスタンスへのLIME の適用結果が表示されます。 

   * - SHAP
     - 対象のインスタンスへのSHAP の適用結果が表示されます。 対象のインスタンスへのSHAP の適用結果が表示されます。 分類において、正の影響を及ぼした箇所が赤色、負の影響を及ぼした箇所が青色として、元画像上に重ねて表示されます。

   * - SmoothGrad
     - 対象のインスタンスへのSmoothGrad の適用結果が表示されます。感度マップを意味するグレースケールの画像として表示されます。 


XAI/XAI Visualization(batch)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

画像分類を行うモデルにおいて、分類結果に影響を及ぼす入力画像の箇所をGrad-CAM、LIME、SHAP、SmoothGradを用いて一度に可視化します。XAI/XAI Visualizationプラグインが1枚の画像に対して処理を行うのに対し、XAI/XAI Visualization(batch)プラグインは指定するデータセットに含まれる複数枚の画像に一括して処理を行います。

| Grad-CAM:
| `Ramprasaath R Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, Dhruv Batra. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization". Proceedings of the IEEE International Conference on Computer Vision, 2017. <https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html>`_
|
| LIME:
| `Marco Tulio Ribeiro, Sameer Singh, Carlos Guestrin. "Why Should I Trust You?: Explaining the Predictions of Any Classifier". Knowledge Discovery and Data Mining, 2016. <https://dl.acm.org/doi/abs/10.1145/2939672.2939778>`_
|
| SHAP:
| `Scott M Lundberg, Su-In Lee. "A unified approach to interpreting model predictions". Proceedings of the 31st International Conference on Neural Information Processing Systems, 2017. <https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html>`_
|
| SmoothGrad:
| `Daniel Smilkov, Nikhil Thorat, Been Kim, Fernanda Viégas, Martin Wattenberg. "Smoothgrad: removing noise by adding noise". Workshop on Visualization for Deep Learning, ICML, 2017. <https://arxiv.org/abs/1706.03825>`_

Input Information
===================

.. list-table::
   :widths: 30 70
   :class: longtable

   * - input
     -
        XAI Visualization の各処理を行う対象の画像ファイル一覧を含むデータセットCSVファイルを指定します。
        
        評価タブの出力結果に含まれる画像データに対しXAI Visualization を行うには、デフォルトのoutput_result.csvを使用します。

   * - model
     -
        各処理の演算に用いるConvolutional Neural Networksのモデルファイル（*.nnp）を指定します。
        
        評価タブで選択中の学習結果を元にXAI Visualization を行うには、デフォルトのresults.nnpを使用します。

   * - input_variable
     - inputで指定したデータセットCSVファイルに含まれる変数より、XAI Visualization 処理対象の画像の変数名を指定します。

   * - label_variable
     - inputで指定したデータセットCSVファイルに含まれる変数より、可視化を行うクラスのIndexの変数名を指定します。

   * - output
     -
        可視化結果を出力するデータセットCSVファイルのファイル名を指定します。
        
        評価タブの評価結果からXAI Visualization を実行した場合、学習結果フォルダに指定したファイル名で可視化結果が保存されます。

   * - num_segments
     - LIME 処理の際に、入力画像を分割するセグメントの数を指定します。

   * - num_segments_2
     - LIME 処理の際に、num_segmentsに分割された領域のうち、可視化するセグメントの数を指定します。

   * - num_samples_lime
     - LIME 処理の際に、入力画像と認識結果の関係をサンプリングする回数を指定します。
　　
   * - num_samples_shap
     - SHAP 処理の際に、入力画像と認識結果の関係をサンプリングする回数を指定します。

   * - num_samples_smoothgrad
     - SmoothGrad 処理の際に、入力画像と認識結果の関係をサンプリングする回数を指定します。

   * - batch_size
     - SHAP 処理の際のbatch_sizeです。

   * - interim_layer
     -
        モデルを構成する層の内、input層以外の層に対してSHAP 処理を行う際に指定します。
        
        input層を0番目として、何番目の層に関して SHAP 処理を行うか指定します。
        
        デフォルトではinput層に対して処理が行われます。

   * - noise_level
     -
        標準偏差を計算するためのノイズレベル(0.0 - 1.0)を指定します。
        
        この標準偏差は SmoothGrad の演算に用いるガウシアンノイズを生成するために用いられます。

Output Information
===================

本プラグインの実行結果は 'output' で指定した名前のCSVファイルとして出力されます。
CSVファイル内の各カラムに関しての情報は以下の通りです（以下のリストに無い名称のカラムは、 '評価' の結果得られる output_result.csvと同様の意味です）。

.. list-table::
   :widths: 30 70
   :class: longtable

   * - Grad-CAM
     - 対象のインスタンスへのGrad-CAMの適用結果が表示されます。 元画像にjet カラーマップを重ねた形で表示され、赤色に近いほど分類結果に強く影響した箇所であることを示します。

   * - LIME
     - 対象のインスタンスへのLIMEの適用結果が表示されます。 

   * - SHAP
     - 対象のインスタンスへのSHAPの適用結果が表示されます。 対象のインスタンスへの本プラグインの適用結果が表示されます。 分類において、正の影響を及ぼした箇所が赤色、負の影響を及ぼした箇所が青色として、元画像上に重ねて表示されます。

   * - SmoothGrad
     - 対象のインスタンスへのSmoothGradの適用結果が表示されます。感度マップを意味するグレースケールの画像として表示されます。 