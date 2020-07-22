# -*- coding: utf-8 -*-
"""
Created on Thu May 28 00:05:53 2020

@author: SC
"""

from __future__ import absolute_import

import cv2
import numpy as np
import os
import os.path as osp
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as keras

# turn uint16 to unit8
def unit16b2uint8(img):
    if img.dtype == 'uint8':
        return img
    elif img.dtype == 'uint16':
        return img.astype(np.uint8)
    else:
        raise TypeError('No such of img transfer type: {} for img'.format(img.dtype))


# standardization - turn 2D to 3D
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


# load images and standardization
def load_images(file_names):
    images = []
    for file_name in file_names:
        img = cv2.imread(file_name, -1)
        img = img_standardization(img)
        img = bgr_to_gray(img)
        img = img.astype('float32')
        images.append(img / 255)
    return images


def load_images_result(file_names):
    images = []
    for file_name in file_names:
        img = cv2.imread(file_name, -1)
        img = img_standardization(img)
        img = bgr_to_gray(img)
        img = np.int16(img > 0)
        images.append(img)
    return images


def bgr_to_gray(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


# unet模型损失函数
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 100) / (K.sum(y_true_f) + K.sum(y_pred_f) + 100)


# unet模型损失函数
def dice_coef_np(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = numpy.sum(y_true_f * y_pred_f)
    return (2. * intersection + 100) / (numpy.sum(y_true_f) + numpy.sum(y_pred_f) + 100)


# unet模型损失函数
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


class myUnet(object):
    def __init__(self, img_rows=512, img_cols=512):
        self.img_rows = img_rows
        self.img_cols = img_cols

class BinaryThresholding:
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, img):
        gray = bgr_to_gray(img)
        # image binarization - 0, 255
        (_, binary_mask) = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)
        # filtering
        binary_mask = cv2.medianBlur(binary_mask, 5)
        connectivity = 4
        _, label_img, _, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity, cv2.CV_32S)
        return label_img


if __name__ == "__main__":
    N = 175
    # segmentor = BinaryThresholding(threshold=110)
    image_path = './dataset1/train'
    result_path = './dataset1/train_GT/SEG'
    image_list = sorted([osp.join(image_path, image) for image in os.listdir(image_path)])[0:N]
    result_list = sorted([osp.join(result_path, result) for result in os.listdir(result_path)])[0:N]

    images = load_images(image_list)
    images = np.array(images)
    images = images.reshape(N, 628, 628, 1)

    results = load_images_result(result_list)
    results = np.array(results)
    results = results.reshape(N, 628, 628, 1)

    images_new = np.empty((N, 256, 256))
    results_new = np.empty((N, 256, 256))
    for i in range(N):
        new_shape = (256, 256)
        images_new[i] = cv2.resize(images[i], new_shape)
        results_new[i] = cv2.resize(results[i], new_shape)
        
    plt.imshow(images_new[0])
    plt.show()
    plt.imshow(images_new[1])
    plt.show()
    plt.imshow(images_new[2])
    plt.show()

    images_new = images_new.reshape(N, 256, 256, 1)
    results_new = results_new.reshape(N, 256, 256)

    plt.imshow(results_new[0])
    plt.show()
    
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        input = layers.Input(shape=(256, 256, 1))
        conv2d_1_1 = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu',
                                   name='conv2d_1_1', kernel_initializer='he_normal')(input)
        conv2d_1_2 = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu',
                                   name='conv2d_1_2', kernel_initializer='he_normal')(conv2d_1_1)
        maxpooling2d_1 = layers.MaxPooling2D(pool_size=2, name='maxpooling2d_1')(conv2d_1_2)

        conv2d_2_1 = layers.Conv2D(128, kernel_size=3, padding='same', activation='relu',
                                   name='conv2d_2_1', kernel_initializer='he_normal')(maxpooling2d_1)
        conv2d_2_2 = layers.Conv2D(128, kernel_size=3, padding='same', activation='relu',
                                   name='conv2d_2_2', kernel_initializer='he_normal')(conv2d_2_1)
        maxpooling2d_2 = layers.MaxPooling2D(pool_size=2, name='maxpooling2d_2')(conv2d_2_2)

        conv2d_3_1 = layers.Conv2D(256, kernel_size=3, padding='same', activation='relu',
                                   name='conv2d_3_1', kernel_initializer='he_normal')(maxpooling2d_2)
        conv2d_3_2 = layers.Conv2D(256, kernel_size=3, padding='same', activation='relu',
                                   name='conv2d_3_2', kernel_initializer='he_normal')(conv2d_3_1)
        maxpooling2d_3 = layers.MaxPooling2D(pool_size=2, name='maxpooling2d_3')(conv2d_3_2)

        conv2d_4_1 = layers.Conv2D(512, kernel_size=3, padding='same', activation='relu',
                                   name='conv2d_4_1', kernel_initializer='he_normal')(maxpooling2d_3)
        conv2d_4_2 = layers.Conv2D(512, kernel_size=3, padding='same', activation='relu',
                                   name='conv2d_4_2', kernel_initializer='he_normal')(conv2d_4_1)
        dropout_4 = layers.Dropout(rate=0.5, noise_shape=None, seed=None, name='dropout_4')(conv2d_4_2)
        maxpooling2d_4 = layers.MaxPooling2D(pool_size=2, name='maxpooling2d_4')(dropout_4)

        conv2d_5_1 = layers.Conv2D(1024, kernel_size=3, padding='same', activation='relu',
                                   name='conv2d_5_1', kernel_initializer='he_normal')(maxpooling2d_4)
        conv2d_5_2 = layers.Conv2D(1024, kernel_size=3, padding='same', activation='relu',
                                   name='conv2d_5_2', kernel_initializer='he_normal')(conv2d_5_1)
        dropout_5 = layers.Dropout(rate=0.5, noise_shape=None, seed=None, name='dropout_5')(conv2d_5_2)

        conv2d_6_1 = layers.Conv2D(512, kernel_size=3, padding='same', activation='relu',
                                   name='conv2d_6_1', kernel_initializer='he_normal')(layers.UpSampling2D(size=(2, 2))(dropout_5))
        concatenate_6 = layers.Concatenate(axis=-1, name='concatenate_6')([dropout_4, conv2d_6_1])
        conv2d_6_2 = layers.Conv2D(512, kernel_size=3, padding='same', activation='relu',
                                   name='conv2d_6_2', kernel_initializer='he_normal')(concatenate_6)
        conv2d_6_3 = layers.Conv2D(512, kernel_size=3, padding='same', activation='relu',
                                   name='conv2d_6_3', kernel_initializer='he_normal')(conv2d_6_2)

        conv2d_7_1 = layers.Conv2D(256, kernel_size=3, padding='same', activation='relu',
                                   name='conv2d_7_1', kernel_initializer='he_normal')(layers.UpSampling2D(size=(2, 2))(conv2d_6_3))
        concatenate_7 = layers.Concatenate(axis=-1, name='concatenate_7')([conv2d_3_2, conv2d_7_1])
        conv2d_7_2 = layers.Conv2D(256, kernel_size=3, padding='same', activation='relu',
                                   name='conv2d_7_2', kernel_initializer='he_normal')(concatenate_7)
        conv2d_7_3 = layers.Conv2D(256, kernel_size=3, padding='same', activation='relu',
                                   name='conv2d_7_3', kernel_initializer='he_normal')(conv2d_7_2)

        conv2d_8_1 = layers.Conv2D(128, kernel_size=3, padding='same', activation='relu',
                                   name='conv2d_8_1', kernel_initializer='he_normal')(layers.UpSampling2D(size=(2, 2))(conv2d_7_3))
        concatenate_8 = layers.Concatenate(axis=-1, name='concatenate_8')([conv2d_2_2, conv2d_8_1])
        conv2d_8_2 = layers.Conv2D(128, kernel_size=3, padding='same', activation='relu',
                                   name='conv2d_8_2', kernel_initializer='he_normal')(concatenate_8)
        conv2d_8_3 = layers.Conv2D(128, kernel_size=3, padding='same', activation='relu',
                                   name='conv2d_8_3', kernel_initializer='he_normal')(conv2d_8_2)

        conv2d_9_1 = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu',
                                   name='conv2d_9_1', kernel_initializer='he_normal')(layers.UpSampling2D(size=(2, 2))(conv2d_8_3))
        concatenate_9 = layers.Concatenate(axis=-1, name='concatenate_9')([conv2d_1_2, conv2d_9_1])
        conv2d_9_2 = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu',
                                   name='conv2d_9_2', kernel_initializer='he_normal')(concatenate_9)
        conv2d_9_3 = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu',
                                   name='conv2d_9_3', kernel_initializer='he_normal')(conv2d_9_2)
        conv2d_9_4 = layers.Conv2D(2, kernel_size=3, padding='same', activation='relu',
                                   name='conv2d_9_4', kernel_initializer='he_normal')(conv2d_9_3)

        conv2d_10 = layers.Conv2D(1, kernel_size=1, activation='sigmoid', name='conv2d_10')(conv2d_9_4)

        model = models.Model(inputs=input, outputs=conv2d_10)

    model.summary()
    model.compile(optimizer=optimizers.SGD(lr=1e-5, momentum=0.9, nesterov=True), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x=images_new, y=results_new, batch_size=16, epochs=16)
    model.save_weights('unet_test_10.h5', save_format='h5')

    new = model.predict(images_new).reshape(N, 256, 256)
    plt.imshow(new[0])
    plt.show()