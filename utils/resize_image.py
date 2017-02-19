# -*- coding: utf-8 -*-
"""
Created on Sat Nov 05 09:52:13 2016

@author: dwipr

Utility code for task number 01 : Resize image to half of it's original image.
Image resized from 1242 x 675 to 621 x187

"""
#%%
# Importing required libraries
from PIL import Image
import os
#%%
TARGET_WIDTH = 1242 
TARGET_HEIGHT = 376
def resize_images(task_type):
    img_input_path = '../input/_temp/raw/training/image_2'
    img_output_path = '../input/_temp/resized/training/image'

    gt_input_path = '../input/_temp/raw/training/gt_image_2'
    gt_output_path = '../input/_temp/resized/training/gt'
    
    test_input_path = '../input/_temp/raw/testing/image_2_um'
    test_output_path = '../input/_temp/resized/test'

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