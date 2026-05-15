# MinkUNeXt-VINE

This repository contains the code of the MinkUNeXt-VINE convolutional neural network. This method is a compressed version of the [MinkUNeXt](https://github.com/juanjo-cabrera/MinkUNeXt) model, which was enhanced with Matryoshka Representation Learning to improve the results with low-cost sparse LiDAR sensors and low-resolution inputs.

Published paper in Ecological Informatics (Elsevier): [Low Cost, High Efficiency: LiDAR Place Recognition in Vineyards with Matryoshka Representation Learning](https://doi.org/10.1016/j.ecoinf.2026.103780)

**Abstract**
Localization in agricultural environments is challenging due to their unstructured nature and lack of distinctive landmarks. Although agricultural settings have been studied in the context of object classification and segmentation, the place recognition task for mobile robots is not trivial in the current state of the art. In this study, we propose MinkUNeXt-VINE, a lightweight, deep-learning-based method that surpasses state-of-the-art methods in vineyard environments thanks to its pre-processing and Matryoshka Representation Learning multi-loss approach. Our method prioritizes enhanced performance with low-cost, sparse LiDAR inputs and lower-dimensionality outputs to ensure high efficiency in real-time scenarios. Additionally, we present a comprehensive ablation study of the results on various evaluation cases and two extensive long-term vineyard datasets employing different LiDAR sensors. The results demonstrate the efficiency of the trade-off output produced by this approach, as well as its robust performance on low-cost and low-resolution input data.

The following image displays the architecture of MinkUNeXt-VINE.
<img src="https://github.com/JudithV/MinkUNeXt-VINE/blob/master/imgs/minkunext-VINE_backbone.png" width=720 height=500 title="MinkUNeXt-VINE's architecture">

# Use MinkUNeXt-VINE

Now, we will provide the instructions needed to execute our method.

## 1. Setup

First, clone the repository and install the necessary dependencies required to execute the scripts:

`pip install -r requirements.txt`

## 2. Generate train and test files
Before training the network, you must generate the training and testing example sets. Our study uses two vineyard datasets, and we provide a specific script for each in the `datasets/pointnetvlad/` directory:

    BLT (blt): Bacchus Long-Term Dataset. No inner file customization is required.

    TEMPO-VINE (vmd): Vineyard Multitemporal Dataset.

### Generate Training Tuples:
Run the corresponding script for your desired dataset:

`# For the BLT dataset:
python3 datasets/pointnetvlad/generate_training_tuples_blt.py`

`# For the TEMPO-VINE dataset:
python3 datasets/pointnetvlad/generate_training_tuples_vmd.py`

Note on TEMPO-VINE Customization: If using the vmd dataset, you can customize the training strategy by commenting/uncommenting the conditionals in the last for loop of the main block. Available strategies include:

- Using "run1" as the training set.
- Following specific target zones as a test set (defined by Uy et al. in PointNetVLAD).
- A "one in, one out" process to divide train and test for every trajectory.
*(Please refer to the "Experimental Setup" section of our paper for more details).*

### Generate Evaluation Files:
Once the training tuples are generated, you must create the .pickle files required for the network's evaluation process.
Bash

`python3 datasets/generate_test_sets.py`

You can customize this file to select which dataset you are generating examples for and, in the case of TEMPO-VINE (vmd), which specific evaluation setup you want to use.


## Train MinkUNeXt-VINE
Before starting the training process, configure your experiment's hyperparameters by editing config/general_parameters.yaml.

Key parameters to check:

- Quantization size
- Maximum radius (for filtering point clouds)
- Dataset selection
- Normalization of the point clouds.
- protocol: *Crucial step*. You must point this parameter to the specific train and test files you generated in the previous step.

Once your configuration is ready, launch the training:
`python3 training/train.py`

## Evaluate MinkUNeXt-VINE
To perform evaluation—including cross-season studies—you will use the pnv_evaluate.py script.

Pre-evaluation Checklist:

- Set Weights: Open `config/general_parameters.yaml` and update the `weights_folder` parameter with the path to the weights from your desired training run.
- Set Target Months: Open `pnv_evaluate.py` and navigate to lines 25 and 27 (`eval_database_files` and `eval_query_files`). Load the desired months to evaluate by referencing the .pickle files you generated with the generate_test_sets.py script.

Run the Evaluation:

`python3 pnv_evaluate.py`

## Acknowledgements
This research work is part of the project PID2023-149575OB-I00 funded by MICIU/AEI/10.13039/501100011033 and by FEDER, UE. It is also part of the project CIPROM/2024/8, funded by Generalitat Valenciana, Conselleria de Educación, Cultura, Universidades y Empleo (program PROMETEO 2025).
