# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 21:55:33 2017

@author: dwipr
"""


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import config
import data_loading
from model import Model

HI = 0



def train(n_epochs, trainData, trainLabelOneHot, validationData, validationLabelOneHot):
    ntrain = len(trainData)
    height = config.height 
    width = config.width
    nrclass = config.nrclass

    #%%
    # Define functions
    x = tf.placeholder(tf.float32, [None, height, width, 3])
    y = tf.placeholder(tf.float32, [None, height, width, nrclass])
    keepprob = tf.placeholder(tf.float32)
    
    # Kernels
    ksize = config.ksize
    fsize = config.fsize
    initstdev = 0.01
    
    initfun = tf.random_normal_initializer(mean=0.0, stddev=initstdev)
    # initfun = None
    
    weights = {
        'ce1': tf.get_variable("ce1", shape = [ksize, ksize, 3, fsize], initializer = initfun) ,
        'ce2': tf.get_variable("ce2", shape = [ksize, ksize, fsize, fsize], initializer = initfun) ,
        'ce3': tf.get_variable("ce3", shape = [ksize, ksize, fsize, fsize], initializer = initfun),
        'ce4': tf.get_variable("ce4", shape = [ksize, ksize, fsize, fsize], initializer = initfun),
        'cd4': tf.get_variable("cd4", shape = [ksize, ksize, fsize, fsize], initializer = initfun),
        'cd3': tf.get_variable("cd3", shape = [ksize, ksize, fsize, fsize], initializer = initfun),
        'cd2': tf.get_variable("cd2", shape = [ksize, ksize, fsize, fsize], initializer = initfun),
        'cd1': tf.get_variable("cd1", shape = [ksize, ksize, fsize, fsize], initializer = initfun),
        'dense_inner_prod': tf.get_variable("dense_inner_prod", shape= [1, 1, fsize, nrclass]
                                           , initializer = initfun) # <= 1x1conv
    }
    biases = {
        'be1': tf.get_variable("be1", shape = [fsize], initializer = tf.constant_initializer(value=0.0)),
        'be2': tf.get_variable("be2", shape = [fsize], initializer = tf.constant_initializer(value=0.0)),
        'be3': tf.get_variable("be3", shape = [fsize], initializer = tf.constant_initializer(value=0.0)),
        'be4': tf.get_variable("be4", shape = [fsize], initializer = tf.constant_initializer(value=0.0)),
        'bd4': tf.get_variable("bd4", shape = [fsize], initializer = tf.constant_initializer(value=0.0)),
        'bd3': tf.get_variable("bd3", shape = [fsize], initializer = tf.constant_initializer(value=0.0)),
        'bd2': tf.get_variable("bd2", shape = [fsize], initializer = tf.constant_initializer(value=0.0)),
        'bd1': tf.get_variable("bd1", shape = [fsize], initializer = tf.constant_initializer(value=0.0))
    }
    
    
    
    #%%
    pred = Model(x, weights, biases, keepprob)
    lin_pred = tf.reshape(pred, shape=[-1, nrclass])
    lin_y = tf.reshape(y, shape=[-1, nrclass])
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(lin_pred, lin_y))
    
    # Class label
    predmax = tf.argmax(pred, 3)
    ymax = tf.argmax(y, 3)
    
    # Accuracy
    corr = tf.equal(tf.argmax(y,3), tf.argmax(pred, 3)) 
    accr = tf.reduce_mean(tf.cast(corr, "float"))
    
    # Optimizer
    optm = tf.train.AdamOptimizer(0.0001).minimize(cost)
    batch_size = 1
    n_epochs = n_epochs
    
    print ("Functions ready")
    #%%
    resumeTraining = True
    with tf.Session() as sess:
        # you need to initialize all variables
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()
        checkpoint = tf.train.latest_checkpoint("model_chekcpoint/")
        print ("checkpoint: %s" % (checkpoint))
        if resumeTraining == False:
            print ("Start from scratch")
        elif  checkpoint:
            print ("Restoring from checkpoint", checkpoint)
            saver.restore(sess, checkpoint)
        else:
            print ("Couldn't find checkpoint to restore from. Starting over.")
        
        for epoch_i in range(n_epochs):
            trainLoss = []; trainAcc = []
            num_batch = int(ntrain/batch_size)+1
            
            for _ in range(num_batch):
                randidx = np.random.randint(ntrain, size=batch_size)
            
                batchData = trainData[randidx]
                batchLabel = trainLabelOneHot[randidx]
                sess.run(optm, feed_dict={x: batchData, y: batchLabel, keepprob: 0.7}) # <== Optm is done here!
                trainLoss.append(sess.run(cost, feed_dict={x: batchData, y: batchLabel, keepprob: 1.}))
                trainAcc.append(sess.run(accr, feed_dict={x: batchData, y: batchLabel, keepprob: 1.}))
            
            # Average loss and accuracy
            trainLoss = np.mean(trainLoss)
            trainAcc = np.mean(trainAcc)
            
            # Run test
            valLoss = sess.run(cost, feed_dict={x: validationData, y: validationLabelOneHot, keepprob: 1.})
            valAcc = sess.run(accr, feed_dict={x: validationData, y: validationLabelOneHot, keepprob: 1.})
            valLoss = 0.0
            valAcc = 0.0
            
            print ("[%02d/%02d] trainLoss: %.4f trainAcc: %.2f valLoss: %.4f valAcc: %.2f" 
                   % (epoch_i, n_epochs, trainLoss, trainAcc, valLoss, valAcc))
            
            # Save snapshot
            if resumeTraining and epoch_i % 10 == 0:
                # Save
                saver.save(sess, 'model_chekcpoint/', global_step = epoch_i)
                
                # Train data
                index = np.random.randint(trainData.shape[0])
                refimg = trainData[index, :, :, :].reshape(height, width, 3)
                batchData = trainData[index:index+1]
                batchLabel = trainLabelOneHot[index:index+1]
                predMaxOut = sess.run(predmax, feed_dict={x: batchData, y: batchLabel, keepprob:1.})
                yMaxOut = sess.run(ymax, feed_dict={x: batchData, y: batchLabel, keepprob:1.})
                gtimg = yMaxOut[0].reshape(height, width)
                errimg = gtimg - predMaxOut[0, :, :].reshape(height, width);
                
                # Plot
#                plt.figure(figsize=(12, 4)) 
#                plt.subplot(2, 2, 1); plt.imshow(refimg); plt.title('Input')
#                plt.subplot(2, 2, 2); plt.imshow(gtimg); plt.title('Ground truth')
#                plt.subplot(2, 2, 3); plt.imshow(predMaxOut[0, :, :].reshape(height, width)); plt.title('[Training] Prediction')
#                plt.subplot(2, 2, 4); plt.imshow(np.abs(errimg) > 0.5); plt.title('Error')
#                plt.show() 
                
    #            # Validation data
    #            index = np.random.randint(testData.shape[0])
    #            batchData = testData[index:index+1]
    #            batchLabel = testLabelOneHot[index:index+1]
    #            predMaxOut = sess.run(predmax, feed_dict={x: batchData, y: batchLabel, keepprob:1.})
    #            yMaxOut = sess.run(ymax, feed_dict={x: batchData, y: batchLabel, keepprob:1.})
    #            refimg = testData[index, :, :, :].reshape(height, width, 3)
    #            gtimg = yMaxOut[0, :, :].reshape(height, width)
    #            errimg = gtimg - predMaxOut[0, :, :].reshape(height, width)
    #            # Plot
    #            plt.figure(figsize=(12, 4)) 
    #            plt.subplot(2, 2, 1); plt.imshow(refimg); plt.title('Input')
    #            plt.subplot(2, 2, 2); plt.imshow(gtimg);  plt.title('Ground truth')
    #            plt.subplot(2, 2, 3); plt.imshow(predMaxOut[0, :, :].reshape(height, width)); plt.title('[Validation] Prediction')
    #            plt.subplot(2, 2, 4); plt.imshow(np.abs(errimg) > 0.5); plt.title('Error')
    #            plt.show()
    return trainLoss, trainAcc


#trainData, trainLabelOneHot = data_loading.load_data("train")
##trainlen = len(trainData)
#
#validationData, validationLabelOneHot = data_loading.load_data("validation")
##testlen = len(testimglist)
#for i in range(2):  
#    tf.reset_default_graph()
#    loss, acc = train(1, trainData, trainLabelOneHot, validationData, validationLabelOneHot)
#    HI = HI+1