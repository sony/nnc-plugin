# SGD Influence
Using a method called SGD Influence, the influence of the input images on recognition result is evaluated. The dataset and the scores are shown in the influential order, which can be referred for data cleansing.
The SGD Influence calculation in this plugin uses an approximate version of algorithm based on the following two papers.

>Data Cleansing for Deep Neural Networks with Storage-efficient Approximation of Influence Functions
Kenji Suzuki, Yoshiyuki Kobayashi, Takuya Narihira
https://arxiv.org/abs/2103.11807

>Data Cleansing for Models Trained with SGD
Satoshi Hara, Atsushi Nitanda, Takanori Maehara
https://arxiv.org/abs/1906.08473

>Understanding Black-box Predictions via Influence Functions
Pang Wei Koh, Percy Liang
https://arxiv.org/abs/1703.04730

# SGD Influence (Tabular)
Using a method called SGD Influence, the influence of the features in input table data on classification result is evaluated. The dataset index and the scores are shown in the influential order, which can be referred for data cleansing. This plugin can be used for models without dropout layer since grad calculation is not available for the moment.

>Data Cleansing for Models Trained with SGD
Satoshi Hara, Atsushi Nitanda, Takanori Maehara
https://arxiv.org/abs/1906.08473

Property | Notes
-- | --
model | Specify the model file (*.nnp) that will be used in the SGD Influence computation. To perform SGD Influence based on the training result selected in the Evaluation tab, use the default results.nnp.
batch_size | Specify the batch size to train with the model used in SGD Influence.
input-train | Specify the dataset CSV file containing the tabular data for which SGD Influence scores are calculated.
input-val |  Specify the dataset CSV file containing the tabular data with which SGD Influence scores are calculated. This input-val dataset are used for SGD Influence scores calculation in accordance with input-train dataset, although the target of scoring are input-train dataset only. Specify the CSV file with different datasets other than input-train.
output | Specify the name of the CSV file to output the inference results to.
seed | Specify the random seed number to shuffle input-train data.
