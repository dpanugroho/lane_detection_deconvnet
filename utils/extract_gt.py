# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 22:15:19 2017

@author: dwipr
"""

from skimage import io
import os

gt_dir = "../input/_temp/padded/training/gt/"
extracted = "../input/_temp/padded_gt_extracted/"

proccessed_dir = gt_dir
for fname in os.listdir(proccessed_dir):
    img = io.imread(proccessed_dir+fname)[:,:,2]
    io.imsave(extracted+fname, img)
