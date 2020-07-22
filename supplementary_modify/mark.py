# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 20:13:52 2020

@author: SC
"""

from __future__ import absolute_import

import cv2
import imageio
import numpy as np
import os
import os.path as osp

image_path = './dataset2/train'
images = sorted([osp.join(image_path, img) for img in os.listdir(image_path) if img.find('.tif') != -1])
visual_path = './dataset2/mark'
if not osp.exists(visual_path):
    os.mkdir(visual_path)
    
for idx, image in enumerate(images):
    img = cv2.imread(image, -1)
    cv2.imwrite(osp.join(visual_path, '{:0>3d}_visual.jpg'.format(idx)), img.astype(np.uint8))