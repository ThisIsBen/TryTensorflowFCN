# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 14:39:50 2017

@author: chiu

"""
import sys
sys.path.append("tf-image-segmentation/")
from tf_image_segmentation.utils.pascal_voc import convert_pascal_berkeley_augmented_mat_annotations_to_png
#convert_pascal_berkeley_augmented_mat_annotations_to_png('D:/VOCtrainval_11-May-2012/benchmark/benchmark_RELEASE')
convert_pascal_berkeley_augmented_mat_annotations_to_png('D:/VOCtrainval_11-May-2012/benchmark')