# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 05:58:09 2017

@author: dwipr
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from train import train


np.random.seed(149)

height = 192
width = 640
nrclass = 2

import data_loading
trainData, trainLabelOneHot = data_loading.load_data("train")
validationData, validationLabelOneHot = data_loading.load_data("validation")

images = np.vstack((trainData, validationData))
labels = np.vstack((trainLabelOneHot, validationLabelOneHot))

indice = np.arange(len(images))
np.random.shuffle(indice)

first_fold_val = np.arange(0,13)
second_fold_val = np.arange(13,26)
third_fold_val = np.arange(26,39)
fourth_fold_val = np.arange(39,53)
fifth_fold_val = np.arange(53,67)
sixth_fold_val = np.arange(67,81)
seventh_fold_val = np.arange(81,95)

cv_split_indices = [first_fold_val, second_fold_val, third_fold_val, 
                    fourth_fold_val, fifth_fold_val, sixth_fold_val,
                    seventh_fold_val]

cv_loss = 0
cv_acc = 0
n_epoch = 1
for i in range(len(cv_split_indices)):
    trainIdx = []
    for j in range(len(cv_split_indices)):
        if j != i:
            trainIdx = np.concatenate((trainIdx, cv_split_indices[j]))
    validationIdx = cv_split_indices[i]

    train_images = images[list(np.asarray(trainIdx, dtype='int')),...]
    validation_images = images[list(np.asarray(validationIdx, dtype='int')),...]

    train_label = labels[list(np.asarray(trainIdx, dtype='int')),...]
    validation_label = labels[list(np.asarray(validationIdx, dtype='int')),...]
                              
    loss, acc = train(n_epochs=1, 
                      trainData=train_images,
                      trainLabelOneHot=train_label,
                      validationData=validation_images,
                      validationLabelOneHot=validation_label)
    
    cv_loss+=loss
    cv_acc+=acc

print("CV Loss", cv_loss)
print ("CV Acc", cv_acc)
