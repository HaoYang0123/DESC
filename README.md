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

## Preprocess data
```bash
#!/bin/bash
python3 preprocess/split_pctr.py  # input train.csv and test.csv, output: pctr_split.json (100 bin pCTR information)
```

## Train DESC model
```bash
#!/bin/bash

set -x

cd DESC

train_path=$1  # train.csv
test_path=$2   # test.csv
pctr_info_path=$3  # pctr_split.json (obtained in preprocess step)
output_model_folder=$4  # output model folder
outpath=$5  # output path (res.csv), add a new column for 'calibrated score'
need_keys='101 121 122 124 125 126 127 128 129 205 206 207 210 216 508 509 702 853 301 109_14 110_14 127_14 150_14' # for AliCCP data
#need_keys='C1 C2 C3 C4 C5 C6 C7 C8 C9 C10 C11 C12 C13 C14 C15 C16 C17 C18 C19 C20 C21 C22 C23 C24 C25 C26'  # for CRETIO data

CUDA_VISIBLE_DEVICES=0 python -u train_desc.py \
    --sample-path ${train_path} \
    --test-sample-path ${test_path} \
    --split-path ${pctr_info_path} \
    --need-keys ${need_keys} \
    --label-name 'click' \
    --pctr-label-name 'pctr' \
    --batch-size 16384 \
    --epoches 1 \
    --workers 0 \
    --learning-rate 1e-3 \
    --model-folder ${output_model_folder} \
    --emb-size 128 \
    --eval-freq 1.1 \
    --dropout 0.2 \
    --lambda-v 1.0 \
    --seed 44 \
    --outpath ${outpath} \
    --fc-hidden-size-str '128,64,32,1'

```
