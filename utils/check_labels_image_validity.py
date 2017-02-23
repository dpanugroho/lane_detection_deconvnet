# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 19:26:06 2017

@author: dwipr
"""

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, img_as_bool

fname = "E:/Computer Science/Computer Vision/lane_detection_deconvnet/input_small_sample/train/gt/um_lane_000000.png"
#img = Image.open(fname)
#img = img.split()[2].convert("1")
#plt.imshow(img)

img2 = img_as_bool(io.imread(fname, as_grey=False)[:,:,2])
io.imshow(img2)