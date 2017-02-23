# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 07:00:42 2017

@author: dwipr
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(149)

height = 192
width = 640
nrclass = 2

import data_loading
trainData, trainLabelOneHot = data_loading.load_data("train")
y = tf.placeholder(tf.float32, [None, height, width, nrclass])
ymax = tf.argmax(y, 3)

ntrain = len(trainData)
batch_size = 1
with tf.Session() as sess:
    randidx = np.random.randint(ntrain, size=batch_size)
    print("randidx",randidx)
    batchData = trainData[randidx]
    batchLabel = trainLabelOneHot[randidx]
    yMaxOut = sess.run(ymax, feed_dict={y: batchLabel})
for i in range(len(yMaxOut)):
    print(i)
    gtimg = yMaxOut[i].reshape(height, width)
    plt.imshow(gtimg)