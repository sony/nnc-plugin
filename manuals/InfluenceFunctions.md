# Influence Functions
Using a method called Influence Functions, the influence of the input images on recognition result are evaluated. The dataset and the scores are shown in the influential order, which can be referred for data cleansing.

Understanding Black-box Predictions via Influence Functions
Pang Wei Koh, Percy Liang
https://arxiv.org/abs/1703.04730

Property | Notes
-- | --
input-train | Specify the dataset CSV file containing image files for which Influence Functions scores are calculated.
input-val |  Specify the dataset CSV file containing image files with which Influence Functions scores are calculated. This input-val dataset are used for Influence Functions scores calculation in accordance with input-train dataset, although the target of scoring are input-train dataset only. Specify the CSV file with different datasets other than input-train.
output | Specify the name of the CSV file to output the inference results to.
seed | Specify the random seed number to shuffle  input-train data.
model | Specify the model file (*.nnp) that will be used in the Influence Functions computation. To perform Influence Functions based on the training result selected in the Evaluation tab, use the default results.nnp.
batch_size | Specify the batch size to train with the model used in Influence Functions.