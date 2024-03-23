## source code https://gist.github.com/Nannigalaxy/35dd1d0722f29672e68b700bc5d44767

import cv2
import os
import pandas as pd 
import dask
import dask.dataframe as dd
import numpy as np
import math
import matplotlib.image as img
import csv


def center_crop(img, dim):
	"""Returns center cropped image
	Args:
	img: image to be center cropped
	dim: dimensions (width, height) to be cropped
	"""
	width, height = img.shape[1], img.shape[0]

	# process crop width and height for max available dimension
	crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
	crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] 
	mid_x, mid_y = int(width/2), int(height/2)
	cw2, ch2 = int(crop_width/2), int(crop_height/2) 
	crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
	return crop_img

def scale_image(img, factor):
	"""Returns resize image by scale factor.
	This helps to retain resolution ratio while resizing.
	Args:
	img: image to be scaled
	factor: scale factor to resize
	"""
	return cv2.resize(img,(int(math.ceil(img.shape[1]*factor)), int(math.ceil(img.shape[0]*factor))))

# source images path to be cropped
PATH = "/home/speedy/Desktop/dpp-test/1"
img_list = os.listdir(PATH)

# writes output file to PATH2 for manual inspection
PATH2 = "/home/speedy/Desktop/dpp-test/output_scaled"
img_list_scaled = os.listdir(PATH2)

# create output folders to save new cropped images, check if exists and if not create
# final output 
if os.path.isdir("output"):
    print("output Exists")
else:
    print("output Doesn't exists")
    os.mkdir("output")
# scaled output    
if os.path.isdir("output_scaled"):
    print("output_scaled exists")
else:
    print("output_scaled Doesn't exists")
    os.mkdir("output_scaled")
    
#set target image size, eg. 64, 128, 256, etc. 
img_size = 192

#set array to store croped images 
img_arr = []

#set matching labels for export to .npy
label_arr = []
label = 1

port_count = 0
land_count = 0
    
## resize to 128 by w or h, whichever is lower
print("Begin for loop!")
for img_f in img_list:
    if img_f.endswith(".png"):
        img = cv2.imread(os.path.join(PATH,img_f))
        h, w, c = img.shape
        if(h >= img_size and w >= img_size):
            h_ratio = img_size / h
            w_ratio = img_size / w  
            if(w_ratio >= h_ratio):
                port_count = port_count + 1
                scaled_img = scale_image(img, w_ratio)
                cropped_img = center_crop(scaled_img, (img_size,img_size))
                # remove below to write cropped image in the output folder (PATH2)
                # cv2.imwrite("output_scaled/" + img_f, cropped_img)
                img_reshape = np.reshape(cropped_img, -1)
                img_arr.append(img_reshape)
                label_arr.append(label)
            else:
                land_count = land_count + 1
                scaled_img = scale_image(img, h_ratio)
                cropped_img = center_crop(scaled_img, (img_size,img_size))
                # remove below to write cropped image in the output folder (PATH2)
                #cv2.imwrite("output_scaled/" + img_f, cropped_img)
                img_reshape = np.reshape(cropped_img, -1)
                img_arr.append(img_reshape)
                label_arr.append(label)

print("Finished for loop!")
cv2.destroyAllWindows()  

## export img array to .npy
print("Begin output to .npy")
# .npy file 
np.save('output.npy', img_arr)
# labels    
np.save('labels.npy', label_arr)

print("Finished output to .npy")
print("Portrait: ", port_count)
print("Landscape: ", land_count)
print("Dataset size: ", land_count + port_count)

## refactored resize and crop into one for loop as above, leaving below for troubleshooting
""" ## crop from scaled img
for img_f in img_list_scaled:
    if img_f.endswith(".png"):
        img = cv2.imread(os.path.join(PATH2,img_f))
        cropped_img = center_crop(img, (128,128))
        cv2.imwrite("output/" + img_f, cropped_img) """

