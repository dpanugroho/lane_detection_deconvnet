# -*- coding: utf-8 -*-
"""
Created on Sat Nov 05 09:52:13 2016

@author: dwipr

Utility code for task number 01 : Resize image to half of it's original image.

"""
#%%
# Importing required libraries
from PIL import Image
import os
#%%
TARGET_WIDTH = 1242 
TARGET_HEIGHT = 375
def resize_images(task_type):
    img_input_path = '../_temp/raw/train/image/'
    img_output_path = '../_temp/resized_1242x375/train/image/'

    gt_input_path = '../_temp/raw/train/gt/'
    gt_output_path = '../_temp/resized_1242x375/train/gt/'
    
    test_input_path = '../_temp/raw/test/'
    test_output_path = '../_temp/resized_1242x375/test/'

    if task_type == 'img':
        input_path = img_input_path
        output_path = img_output_path
    elif task_type == 'gt':
        input_path = gt_input_path
        output_path = gt_output_path
    elif task_type == 'test':
        input_path = test_input_path
        output_path = test_output_path
    else :
        print('Task type shuld be either "gt" or "img"')
        return
    
    files = os.listdir(input_path)
    for filename in files:
        # choosing only um* images and um_lane gt_images
        if(((task_type == 'img' or task_type=='test') and 
        filename.split('_')[0] == 'um') or (task_type == 'gt' and 
        filename.split('_')[1] == 'lane')):
            print(filename)
            
            
            curr_img = Image.open(os.path.join(input_path, filename), 'r')
            
            resized_img = curr_img.resize((TARGET_WIDTH,TARGET_HEIGHT), Image.ANTIALIAS)
            resized_img.save(os.path.join(output_path, filename))

#%%
# Main code execution

if __name__ == "__main__":
    resize_images('img')
    resize_images('gt')
    resize_images('test')
    
#%%
#curr_img = Image.open("E:/Computer Science/Skripsi/test_real.PNG", 'r')
#
#resized_img = curr_img.resize((621,187), Image.ANTIALIAS)
#resized_img.save("E:/Computer Science/Skripsi/test_real.PNG")
##%%