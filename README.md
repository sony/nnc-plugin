# Plugins for Neural Network Console \(English/[日本語](README_ja.md)\)

![](./img/plugin.png)

## What is the plugins for Neural Network Console ?
The plugins enable pre-processing and post-processing on Neural Network Console. There are already plugins on Neural Network Console, but you can use the latest release of the pulings from this repository.


## Current lineup of the plugins

###  Dataset
* Fairness
    * [Demographic Parity Difference](./manuals/Dataset/Fairness/demographic_parity.rst)
    * [Disparate Impact](./manuals/Dataset/Fairness/disparate_impact.rst)

###  Pre-processing
* Create dataset
    * [Object detection (from Yolo v2 format)](./manuals/Pre_Process/Create_Dataset/ObjectDetection.rst)
    * [Object detection (for CenterNet from Yolo v2 format)](./manuals/Pre_Process/Create_Dataset/ObjectDetection_CenterNet.rst)
    * [Simple  (Japanese) text classification](./manuals/Pre_Process/Create_Dataset/SimpleTextClassification.rst)
    * [String classification](./manuals/Pre_Process/Create_Dataset/StringClassification.rst)
    * [Simple tabular data](./manuals/Pre_Process/Create_Dataset/SimpleTabularDataset.rst)
    * [Split image](./manuals/Pre_Process/Create_Dataset/SplitImage.rst)
    * [Split wav](./manuals/Pre_Process/Create_Dataset/SplitWav.rst)

###  Post-processing
* Utils
    * [Cross tabulation](./manuals/Post_Process/Utils/CrossTabulation.rst)
    * [CSV to wav](./manuals/Post_Process/Utils/CSVtoWAV.rst)
    * [Inference](./manuals/Post_Process/Utils/Inference.rst)
    * [Parameter stats](./manuals/Post_Process/Utils/ParameterStats.rst)
    * [Restore split images / wav](./manuals/Post_Process/Utils/RestoreSplitImageWav.rst)
    * [Similar words](./manuals/Post_Process/Utils/SimilarWords.rst)
    * [Simple (Japanese) text generation](./manuals/Post_Process/Utils/SimpleTextGeneration.rst)
    * [tSNE](./manuals/Post_Process/Utils/tSNE.rst)
    
* Visualization
    * [Scatter plot](./manuals/Post_Process/Visualization/ScatterPlot.rst)
    * [Tile images](./manuals/Post_Process/Visualization/TileImages.rst)

* eXplainable AI (XAI)
    * [Attention Editor](./manuals/Post_Process/XAI/AttentionEditor.rst)
    * [SGD influence](./manuals/Post_Process/XAI/SGDInfluence.rst)
    * [Influence Functions](./manuals/Post_Process/XAI/InfluenceFunctions.rst)
    * [Face evaluation](./manuals/Post_Process/XAI/FaceEvaluation.rst)
    * [Grad-CAM](./manuals/Post_Process/XAI/GradCAM.rst)
    * [LIME](./manuals/Post_Process/XAI/LIME.rst)
    * [SHAP](./manuals/Post_Process/XAI/SHAP.rst)
    * [Smooth Grad](./manuals/Post_Process/XAI/SmoothGrad.rst)
    * [TracIn](./manuals/Post_Process/XAI/TracIn.rst)
    * [RepresenterPoint](./manuals/ja/Post_Process/XAI/Representerpoint.rst)
    * [Attention Map Visualization](./manuals/Post_Process/XAI/AttentionMapVisualization.rst)
    
* Fairness
    * [CV Score](./manuals/Post_Process/Fairness/calders_verwer_score.rst)
    * [Demogrphic Parity](./manuals/Post_Process/Fairness/demographic_parity.rst)
    * [Disparate Impact](./manuals/Post_Process/Fairness/disparate_impact.rst)
    * [Equal Opportunity](./manuals/Post_Process/Fairness/equal_opportunity.rst)
    * [Equalised Odds](./manuals/Post_Process/Fairness/equalised_odd.rst)
    * [Theil Index](./manuals/Post_Process/Fairness/theil_index.rst)
    * [Reweighing](./manuals/Post_Process/Fairness/Reweighing.rst)

## How to use the latest plugins
The plugins run on Neural Network Console. If you do not have Neural Network Console, please download from here (https://dl.sony.com/).
1. Download the zip files from this repository. 
2. Extract the zip files on your PC.
3. Delete the existing plugins folder. You can find it from neural_network_console > libs > **plugins**. 
* **NOTE** If you do not want to turn off some plugins, please leave them.
4. Put the downloaded **plugins** folders in the same place, neural_network_console > libs > **plugins**.  
5. Restart Neural Network Console.

###  Pre-processing
* To execute the plugins of the pre-processing, select the "DATASET" on the left of the top screen. Then  click "Create Dataset", you can select the plugins of the pre-processing.
<p align="center">
<img src="./img/Preprocessing.png" width="400px">  
</p>


### Post-processing
* To execute the plugins of the post-processing, right-click the evaluation results on the Evaluation tab to open a shortcut menu and select the plugins.
<p align="center">
<img src="./img/postprocessing.png" width="400px">  
</p>

## Sample projects
### XAI
* [Attention branch network](./samples/xai) resnet110-attention-branch-network.sdcproj
* [TracIn](./samples/xai) resnet56-tracin.sdcproj
* [RepresenterPoint](.\samples\xai\README.md) vgg16-representer-point.sdcproj

### Fairness
* [Prejudice Remover Regularizer](./samples/fairness/prejudice-remover-regularizer/README.md) prejudice_remover_regularizer.sdcproj

### How to use the latest sample projects
The sample projects run on Neural Network Console. 
1. Download the zip files from this repository. 
2. Extract the zip files on your PC.
3. You can place the downloaded **sample projects** anywhere, but it must be a folder without double-byte characters.
