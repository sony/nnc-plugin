Fairness/Reweighing
~~~~~~~~~~~~~~~~~~~~~~~~

This plugin pre-processes data to reweighing value of the instances.

`Kamiran, Faisal and Calders, Toon. "Data preprocessing techniques for classification without discrimination". Knowledge and Information Systems, 33(1):1-33, 2012. <https://link.springer.com/article/10.1007/s10115-011-0463-8>`_

Input Information
===================

.. list-table::
   :widths: 30 70
   :class: longtable
   :header-rows: 1

   * - Property
     - Notes

   * - input-train
     - Specify the dataset CSV file.

   * - label-name
     - Of the variables included in the dataset CSV file specified in input, specify the variable name of the target label to be analyzed.

   * - protected-attribute
     - Of the variables included in the dataset CSV file specified in input, specify the variable names of interest that is considered to convey bias to the result.

   * - output
     - Specify the name of the CSV file to output the results to.

Output Information
===================

The result of this plugin is saved in the designated 'output' path as CSV file.
The information on the columns of CSV file is as follows.

.. list-table::
   :widths: 30 70
   :class: longtable

   * - Sample_weight
     - The sample weight of the instances based on protected-attribute.

Example Use
===================

Download the '_Fairness' folder the nnc-plugin repository, Place 'Fairness' folder under 'neural_network_console\libs\plugins\_Post_Process'.
Click 'Open Dataset' in Neural Network Console, Select 'german_credit.sdcproj' under 'neural_network_console\libs\plugins\_Post_Process\_Fairness\bias_mitigation_utils'.
Right-click on 'german_credit.sdcproj' on the project on the home screen, select in the order of 'Other tools', 'Command prompt', Type 'python create_german_credit_csv.py' at the command prompt to run.
Restart the Neural Network Console.
Let us use german_credit.sdcproj in Neural Network Console sample project to have an example use of this plugin.
The output result of project is shown after train and evaluation.
Right click somewhere on the output result to select 'Plugin', 'Fairness' and then 'bias mitigation' to open the window of the plugin for condition select.
The attributes of 'input-train' is automatically filled in.
The attributes of 'label-name' and 'protected-attribute' allow column label names in input-train dataset CSV file.
Use 'y__0:Good / bad' for 'label-name' and 'x__32:Personal status and sex=A91' for 'protected-attribute'.
This protected attribute contains information of gender (as written sex=A91).
It is not favorable for the feature to be judged 'y__0:Good / bad' in credit risk evaluation.
This plugin gives output result in 'output' CSV file with reweighing value of instances.

