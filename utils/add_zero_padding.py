
    # -*- coding: utf-8 -*-
"""
Created on Mon Nov 07 20:07:23 2016

@author: dwipr

Add Reflection/Mirror padding to Images
"""

# Import library yang dibutuhkan
from PIL import Image
import os
#%% 

# Function untuk melakukan padding terhadap satu citra
def pad_one_image(path, fname):
    cur_img = Image.open(os.path.join(path, fname))
    input_img_size=cur_img.size
    padded_size = (1280, 384)
    padded_img = Image.new("RGB", padded_size)   
    padded_img.paste(cur_img, 
                     (int((padded_size[0]-input_img_size[0])/2),
                      int((padded_size[1]-input_img_size[1])/2)))    
    return padded_img
    
#%%
# Function untuk melakukan padding ke banyak citra sekaligus
def padd_images(task_type):
    img_input_path = '../_temp/resized/train/image/'
    img_output_path = '../_temp/padded/train/image/'

    gt_input_path = '../_temp/resized/train/gt/'
    gt_output_path = '../_temp/padded/train/gt/'

    test_input_path = '../_temp/resized/test/'
    test_output_path = '../_temp/padded/test/'

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
        # padding image
        print(filename)
        padded_img = pad_one_image(input_path, filename)
        padded_img.save(os.path.join(output_path, filename))
        
#%%
# Main code execution

if __name__ == "__main__":
    padd_images('img')
    print('Finished padding images')
    padd_images('gt')
    print('Finished padding ground truths')
    padd_images('test')
    print('Finised padding tests')
    
