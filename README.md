# monkey_hrnet

## Introduction
This is a simple application of HRNet to do pose estimation on monkey videos. The model was slightly modified with deconvolutional layers before the final layer of HRNet. The training dataset was created by DeepLabCut's labeling GUI. 
 
## Installation
Follow [HRNet](https://github.com/HRNet/HRNet-Human-Pose-Estimation#quick-start)'s installation guide.

## Results
For each test settings, several likelihood, maximum value of heatmap, thresholds were tested.

DLC (not trained with low-resolution images):
Likelihood cutoff | Mean Err (pixels) | Std Err (pixels) | Proportion of rejection
------------ | ------------- | ------------ | -------------
0.05 | 17.36 | 0.00621 | 0.043
0.10 | 17.22 | 0.00639 | 0.090
0.15 | 17.35 | 0.00692 | 0.163
0.20 | 17.53 | 0.00740 | 0.218
0.25 | 17.54 | 0.00779 | 0.261
0.30 | 17.45 | 0.00804 | 0.296
0.35 | 17.56 | 0.00846 | 0.329
0.40 | 17.63 | 0.00886 | 0.359
0.45 | 17.82 | 0.00952 | 0.397
0.50 | 17.79 | 0.01018 | 0.438

DLC (trained with low-resolution images):
Likelihood cutoff | Mean Err (pixels) | Std Err (pixels) | Proportion of rejection
------------ | ------------- | ------------ | -------------
0.05 | 6.39 | 0.00171 | 0.000
0.10 | 6.39 | 0.00171 | 0.000
0.15 | 6.39 | 0.00171 | 0.000
0.20 | 6.39 | 0.00171 | 0.000
0.25 | 6.39 | 0.00171 | 0.000
0.30 | 6.39 | 0.00171 | 0.000
0.35 | 6.39 | 0.00171 | 0.000
0.40 | 6.39 | 0.00171 | 0.000
0.45 | 6.39 | 0.00171 | 0.000
0.50 | 6.39 | 0.00171 | 0.000

HRNet only:
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

HRNet+SAM:
Likelihood cutoff | Mean Err (pixels) | Std Err (pixels) | Proportion of rejection
------------ | ------------- | ------------ | -------------
0.05 | 5.78 | 0.00070 | 0.000
0.10 | 5.78 | 0.00070 | 0.000
0.15 | 5.78 | 0.00070 | 0.000
0.20 | 5.78 | 0.00070 | 0.000
0.25 | 5.77 | 0.00070 | 0.004
0.30 | 5.68 | 0.00072 | 0.035
0.35 | 5.60 | 0.00075 | 0.082
0.40 | 5.49 | 0.00080 | 0.143
0.45 | 5.43 | 0.00090 | 0.235
0.50 | 5.27 | 0.00091 | 0.343
