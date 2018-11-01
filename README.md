# Enhanced LSTM for Natural Language Inference
Implementation of the ESIM model for natural language inference with Tensorflow

This repository contains an implementation with Tensorflow of the sequential model presented in the paper ["Enhanced LSTM for Natural Language Inference"](http://www.aclweb.org/anthology/P17-1152) by Chen et al. in 2017.

# Dependencies
Python 2.7 <br>
Tensorflow 1.4.0

# Running the scripts
## Download and preprocess
```
cd data
bash fetch_and_preprocess.sh
```

## Train and test a new model
```
cd scripts
bash train.sh
```
The training process and results are in log.txt file.

## Test a trained model
```
bash test.sh
```
The test results are in log_test.txt file.
