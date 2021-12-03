# -*- coding: utf-8 -*-
""" CIS6930TML -- Homework 2 -- attacks.py

# This file contains the attacks
"""

import os
import sys

import numpy as np
import keras

import sklearn.metrics as metrics
from sklearn.linear_model import LogisticRegression

import scipy.stats as stats


"""
## Compute attack performance metrics, i.e., accuracy and advantage (assumes baseline = 0.5)
## Note: also returns the full confusion matrix
"""
def attack_performance(in_or_out_test, in_or_out_pred):
    cm = metrics.confusion_matrix(in_or_out_test, in_or_out_pred)
    accuracy = np.trace(cm) / np.sum(cm.ravel())
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    advantage = tpr - fpr

    return accuracy, advantage, cm

"""
## Extract a random subdataset of 'sz' records
"""
def random_subdataset(x, y, sz):
    assert x.shape[0] == y.shape[0]
    perm = np.random.permutation(x.shape[0])
    perm = perm[0:sz]

    return x[perm,:].copy(), y[perm,:].copy()



"""
## Perform the posterior attack
## Inputs:
##  - x_targets, y_targets: records to attack
##  - query_target_model: function to query the target model [invoke as: query_target_model(x)]
##  - threshold: decision threshold

##  Output:
##  - in_or_out_pred: in/out prediction for each target
"""
def do_posterior_attack(x_targets, y_targets, query_target_model, threshold=0.9):

    ## TODO ##
    ## Insert your code here
    pv = query_target_model(x_targets)
    
    p = np.argmax(pv, axis = 1)
    
    in_or_out_pred = np.zeros((pv.shape[0],))
    for i in range(0, pv.shape[0]):
        if ( p[i] > threshold):
            in_or_out_pred[i] = 1
        else:
            in_or_out_pred[i] = 0
    return in_or_out_pred
