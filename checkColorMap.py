# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 21:35:22 2017

@author: chiu
"""


from __future__ import division

import os
import sys
import tensorflow as tf
import skimage.io as io
import numpy as np

sys.path.append("tf-image-segmentation/")

#sys.path.append("/models/slim")
sys.path.append("models/slim/")

#fcn_16s_checkpoint_path = 'TrainedModel\fcn_8s\fcn_8s_checkpoint\model_fcn8s_final.ckpt'
fcn_16s_checkpoint_path = \
    'TrainedModel/fcn_8s/fcn_8s_checkpoint/model_fcn8s_final.ckpt'
#os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

slim = tf.contrib.slim

from tf_image_segmentation.models.fcn_8s import FCN_8s

from tf_image_segmentation.utils.inference import adapt_network_for_any_size_input
from tf_image_segmentation.utils.pascal_voc import pascal_segmentation_lut

number_of_classes = 21
pascal_voc_lut = pascal_segmentation_lut()
class_labels= list(pascal_voc_lut)
valid_entries_class_labels = class_labels[:-1]
print(valid_entries_class_labels)