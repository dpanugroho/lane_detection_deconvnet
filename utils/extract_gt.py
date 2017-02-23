# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 22:15:19 2017

@author: dwipr
"""

from PIL import Image
import os

gt_dir = "../input_full/train/gt/"
extracted = "../input_full/gt_train_extracted/"

proccessed_dir = gt_dir
for fname in os.listdir(proccessed_dir):
    img = Image.open(proccessed_dir+fname).convert("1")
    img.save(extracted+fname)
