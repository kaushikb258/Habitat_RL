import numpy as np
import glob
import sys
import PIL
from PIL import Image


RGB_PATH = '/home/kb/habitat_experiments/code/imgs_Quantico/rgb/*.npy'
DEPTH_PATH = '/home/kb/habitat_experiments/code/imgs_Quantico/depth/*.npy'


def get_rgb():
   
    img_paths = glob.glob(RGB_PATH)

    N = len(img_paths)

    d = np.zeros((N, 256, 256, 3),dtype=np.uint8)
 

    for i in range(N): 
       img = np.load(img_paths[i])
       assert img.shape == (256, 256, 3)   
       d[i,:,:,:] = img      

    assert d.shape == (N, 256, 256, 3)

    return d
 

#-------


def get_depth():
   
    img_paths = glob.glob(DEPTH_PATH)

    N = len(img_paths)

    d = np.zeros((N, 256, 256),dtype=np.float32)
 

    for i in range(N): 
       img = np.load(img_paths[i])
       assert img.shape == (256, 256)   
       d[i,:,:] = img      
      
    d = np.expand_dims(d,3)

    assert d.shape == (N, 256, 256, 1)

    return d

#-------


