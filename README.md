# SVM-Spam-SMS-classifier

## Introduction
This repo is an implementation of SVM (Support Vector Machine) classifier to classify spam message and ham message. Also, a Naive Bayes classifier is included as a comparison. 

This classifier contains the model training process, hence it will take some time to train the model for classification.

## Environment
1. Python 3.6
2. `Numpy`
3. `Sklearn`

## How to run it?
Clone this repo and run the `classifier_main.py` in terminal. The training process and test process will start automatically. The statistics will be shown after whole process is completed.

Sample:
`python3 classifier_main.py`

## Data source
There are two databases contains in this repo. One is **UCI Machine Learning Database** marked as `ham` and `spam` in repo. The other one is **Lingspam** and marked as `ham1` and `spam1`.

## To do list
- [ ] Save trained model
- [ ] More data
- [ ] Change word extract mode from each word to some frequent “key word” to reduce time
