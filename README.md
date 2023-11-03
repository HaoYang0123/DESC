## Deep Ensemble Shape Calibration

Deep Ensemble Shape Calibration: Multiple Fields Post-hoc Calibration in Online Advertising


## Introduction
We proposed an new DESC method for calibration on CTR prediction task in [Shopee](https://shopee.co.id/). 


## Requirements and Installation
We recommended the following dependencies.

* Python 3.8
* [PyTorch](http://pytorch.org/) 1.8.0
* Details shown in requirements.txt


## Download public data
1. CRETIO data set can be downloaded from this [link](https://www.kaggle.com/c/criteo-display-ad-challenge).
2. AliCCP data set can be downloaded from this [link](https://tianchi.aliyun.com/datalab/dataSet.html?dataId=408).
3. Industrial data set will be published soon.

## Non-calibrated model
We use DeepFM to train the non-calibrated models with all fields in training data set. DeepFM code can be downloaded from [here](https://github.com/shenweichen/DeepCTR-Torch).
After training DeepFM model, we use this model to predict the non-calibrated scores for all samples, including the training, validation and test data.

## Train DESC model
```bash
#!/bin/bash

set -x

cd DESC

train_path=$1  # train.csv
test_path=$2   # test.csv

CUDA_VISIBLE_DEVICES=0 python -u train_desc.py \
    --sample-path ${train_path} \
    --test-sample-path ${test_path} \
    --split-path /data/apple.yang/Calibration/data/pctr_split.json \
    --need-keys '101 121 122 124 125 126 127 128 129 205 206 207 210 216 508 509 702 853 301 109_14 110_14 127_14 150_14' \
    --label-name 'click' \
    --pctr-label-name 'pctr' \
    --cross-path /data/apple.yang/Calibration/data/feats2_noUse.txt \
    --ts-weight-folder /data/apple.yang/Calibration/data/basic_curves_weights \
    --batch-size 16384 \
    --epoches 1 \
    --workers 0 \
    --learning-rate 1e-3 \
    --model-folder /data/apple.yang/Calibration/data/models \
    --emb-size 128 \
    --eval-freq 1.1 \
    --dropout 0.2 \
    --lambda-v 1.0 \
    --seed 44 \
    --outpath /data/apple.yang/Calibration/data/res3.csv \
    --fc-hidden-size-str '128,64,32,1'

```
