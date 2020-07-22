# -*- coding: utf-8 -*-
from __future__ import absolute_import

import cv2
import numpy as np
import os
import os.path as osp
import pdb
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.optimizers as optimizers
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def unit16b2uint8(img):
    if img.dtype == 'uint8':
        return img
    elif img.dtype == 'uint16':
        return img.astype(np.uint8)
    else:
        #raise TypeError('No such of img transfer type: {} for img'.format(img.dtype))
        return img.astype(np.uint8)

def img_standardization(img):
    img = unit16b2uint8(img)
    if len(img.shape) == 2:
        img = np.expand_dims(img, 2)
        img = np.tile(img, (1, 1, 3))
        return img
    elif len(img.shape) == 3:
        return img
    else:
        raise TypeError('The Depth of image large than 3 \n')
        
def adjustData(img,mask,flag_multi_class,num_class):
    if(flag_multi_class):
        img = img / 255
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            #for one pixel in the image, find the class in mask and convert it into one-hot vector
            #index = np.where(mask == i)
            #index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
            #new_mask[index_mask] = 1
            new_mask[mask == i,i] = 1
        new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
    elif(np.max(img) > 1):
        img = unit16b2uint8(img)
        mask = unit16b2uint8(mask)
        img = img / 255
        mask = mask /255
        mask[mask > 0] = 1
        mask[mask <= 0] = 0
    return (img,mask)

def trainGenerator(batch_size,train_path,mask_path, image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256,256),seed = 1):
    
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        mask_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        yield (img,mask)

def testGenerator(test_path,num_image = 30,target_size = (256,256),flag_multi_class = False,as_gray = True):
    for i in range(num_image):
        img = cv2.imread(os.path.join(test_path,"%d.png"%i),-1)
        #img = unit16b2uint8(img)
        img = img / 255
        img = cv2.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        yield img
        


def origin_main():
    segmentor = BinaryThresholding(threshold=110)
    x_train_path = './dataset1/train/'
    y_train_path = './dataset1/train_GT/SEG/'
    test_path = './dataset1/test/'
    result_path = './dataset1/test_RESRES'
    if not osp.exists(result_path):
        os.mkdir(result_path)
    '''
    x_train_list = sorted([osp.join(x_train_path, x) for x in os.listdir(x_train_path)])
    y_train_list = sorted([osp.join(y_train_path, y) for y in os.listdir(y_train_path)])
    X_train = load_images(x_train_list)
    Y_train = load_images(y_train_list)
    '''
    test_list = sorted([osp.join(test_path, test) for test in os.listdir(test_path)])
    images = load_images(test_list)
    for index, image in enumerate(images):
        label_img = segmentor(image)
        cv2.imwrite(osp.join(result_path, 'mask{:0>3d}.tif'.format(index)), label_img.astype(np.uint16))
        
def IndexInRange(i, bond = 628):
    return (i < bond) and (i >= 0)

def BFS(result, i, j, color):
    result[i, j] = color
    if (IndexInRange(i - 1) and result[i - 1, j] == 255):
        BFS(result, i - 1, j, color)
    if (IndexInRange(i + 1) and result[i + 1, j] == 255):
        BFS(result, i + 1, j, color)
    if (IndexInRange(j - 1) and result[i, j - 1] == 255):
        BFS(result, i, j - 1, color)
    if (IndexInRange(j + 1) and result[i, j + 1] == 255):
        BFS(result, i, j + 1, color)
    return