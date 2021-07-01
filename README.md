# monkey_hrnet

## Introduction
This is a simple application of HRNet to do pose estimation on monkey videos. The model was slightly modified with deconvolutional layers before the final layer of HRNet. The training dataset was created by DeepLabCut's labeling GUI. 
 
## Installation
Follow [HRNet](https://github.com/HRNet/HRNet-Human-Pose-Estimation#quick-start)'s installation guide.

## Results
Several likelihood, maximum value of heatmap, thresholds were tested.
Likelihood cutoff | Mean Err (pixels) | Std Err (pixels) | Proportion of rejection
------------ | ------------- | ------------ | -------------
0.05 | 6.52 | 0.000143 | 0.000
0.10 | 6.52 | 0.000143 | 0.000
0.15 | 6.52 | 0.000143 | 0.001
0.20 | 6.51 | 0.000144 | 0.006
0.25 | 6.46 | 0.000146 | 0.029
0.30 | 6.37 | 0.000157 | 0.081
0.35 | 6.34 | 0.000173 | 0.143
0.40 | 6.28 | 0.000191 | 0.209
0.45 | 6.32 | 0.000226 | 0.296
0.50 | 6.05 | 0.000254 | 0.418
