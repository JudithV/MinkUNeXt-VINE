# MinkUNeXt-VINE

This repository contains the code of the MinkUNeXt-VINE convolutional neural network. This method is a compressed version of the [MinkUNeXt](https://github.com/juanjo-cabrera/MinkUNeXt) model, which was enhanced with Matryoshka Representation Learning to improve the results with low-cost sparse LiDAR sensors and low-resolution inputs.

Preprint: [[2601.18714] Low Cost, High Efficiency: LiDAR Place Recognition in Vineyards with Matryoshka Representation Learning](https://arxiv.org/abs/2601.18714)

**Abstract**
Localization in agricultural environments is challenging due to their unstructured nature and lack of distinctive landmarks. Although agricultural settings have been studied in the context of object classification and segmentation, the place recognition task for mobile robots is not trivial in the current state of the art. In this study, we propose MinkUNeXt-VINE, a lightweight, deep-learning-based method that surpasses state-of-the-art methods in vineyard environments thanks to its pre-processing and Matryoshka Representation Learning multi-loss approach. Our method prioritizes enhanced performance with low-cost, sparse LiDAR inputs and lower-dimensionality outputs to ensure high efficiency in real-time scenarios. Additionally, we present a comprehensive ablation study of the results on various evaluation cases and two extensive long-term vineyard datasets employing different LiDAR sensors. The results demonstrate the efficiency of the trade-off output produced by this approach, as well as its robust performance on low-cost and low-resolution input data.

The following image displays the architecture of MinkUNeXt-VINE.
<img src="https://github.com/JudithV/MinkUNeXt-VINE/blob/master/imgs/minkunext-VINE_backbone.png" width=720 height=500 title="MinkUNeXt-VINE's architecture">

## Setup

Now, we will provide the instructions needed to execute our method.

Firstly, the dependencies required for the executions of our scripts must be installed. This is done with the following command:
`pip install -r requirements.txt`



## Acknowledgements
This research work is part of the project PID2023-149575OB-I00 funded by MICIU/AEI/10.13039/501100011033 and by FEDER, UE. It is also part of the project CIPROM/2024/8, funded by Generalitat Valenciana, Conselleria de Educación, Cultura, Universidades y Empleo (program PROMETEO 2025).
