# PARADISE

This repository is the official implementation of 
"Accurate and Interpretable Decomposition for Temporal Irregular Tensors with Missing Values" (KDD 2026).
<p align="center">
  <img src="https://github.com/anonymousPARADISE/PARADISE/blob/main/docs/overview.png"/>
</p>


## Abstract

Given a temporal irregular tensor with missing values, how can we accurately decompose the tensor and obtain interpretable latent factors? 
Many real-world datasets are represented as temporal irregular tensors, consisting of matrices with shared columns but varying rows across time. 
PARAFAC2 decomposition is widely used to analyze temporal irregular tensors. 
They also handle missing values by optimizing the loss only on observed entries and updating the factors in a row-wise manner. However, there are still two challenges.
First, the temporal factors obtained from PARAFAC2 are hard to interpret, as long-term trends and seasonal patterns are entangled into a single latent factor. 
Second, they update the factors row by row to improve accuracy, which incur high computational costs due to repeated matrix inversions.

In this paper, we propose PARADISE, an accurate and interpretable decomposition method for temporal irregular tensors with missing data. 
PARADISE explicitly separates the temporal latent factor into trend and seasonal components, and enforces their disentanglement through regularization. 
Furthermore, we accelerate row-wise updates via Cholesky decomposition, significantly reducing computational costs. 
As a result, PARADISE achieves up to 10.8% improvement in missing value prediction over existing methods and reduces runtime by 24.7% compared to methods. 
We also analyze the learned factors through case studies, which demonstrate clear disentanglement of temporal dynamics.

## Code Information
All codes are written by MATLAB R2025a.

### Library
We need the following library to run our proposed method.
Please refer to the following, too.
 - `Parallel Computing Toolbox`
 - `Signal Processing Toolbox`
 - `Statistics and Machine Learning Toolbox`
 - `Tensor Toolbox v3.0`
 > * Reference: B. W. Bader, T. G. Kolda et al., “Matlab tensor toolbox version 3.0-dev,” Available online, Oct. 2017. [Online]. Available: <https://www.tensortoolbox.org>

After you download `Tensor Toolbox v3.0`, you should put the directory of `Tensor Toolbox v3.0` in `library` directory.
Without the Tensor Toolbox library, PARADISE is not run.


## Prerequisites
Our code requires Tensor Toolbox (available at https://gitlab.com/tensors/tensor_toolbox).

## Datasets
The datasets are available at [PEMS-SF](https://archive.ics.uci.edu/dataset/204/pems+sf), [PEMS-BAY](https://www.kaggle.com/datasets/scchuy/pemsbay), [VicRoads](https://github.com/florinsch/BigTrafficData), [METR-LA](https://www.kaggle.com/datasets/annnnguyen/metr-la-dataset), [Electricity](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014) and [Weather](https://www.kaggle.com/datasets/shivamshinde1904/weather-data2000-2023).

| Dataset      | Max Dim. I_k     | Dim. J               | Dim. K        | # of nnz | Summary |
|-------------|---------------|--------------------------|---------------------|------|-------|
| **PEMS-SF**  | 440  | 144 | 963 | 34.6M | Traffic |
| **PEMS-BAY**  | 180  | 288 | 325 | 16.8M | Traffic |
| **VicRoads**  | 2033  | 96 | 1084 | 109.7M | Traffic |
| **METR-LA**  | 119  | 288 | 207 | 6.8M | Traffic |
| **Electricity**  | 1460  | 96 | 370 | 41.9M | Electricity |
| **Weather**  | 696  | 7 | 85 | 295.3K | Climate |

## How to Run
You run MATLAB, and type the following commands in MATLAB.

Before you run our proposed method, you should add paths into MATLAB environment. Please type the following command in MATLAB:
```
run addPaths
```
Then, type the following command to run the demo:
```
run main.m
```
Download the demo dataset from [here](https://drive.google.com/file/d/1pG0QKFq_2B-rBSPZrT0sz8R_TeSKYJjM/view?usp=drive_link).
