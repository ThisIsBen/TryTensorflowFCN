# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 15:43:49 2017

@author: chiu
"""

import os, sys
from PIL import Image

sys.path.append("tf-image-segmentation/")
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

#pascal_root = 'D:\\VOCtrainval_11-May-2012\\VOCdevkit\\VOC2012'

pascal_root = 'D:\\TryTensorflowFCN\\3DBuilderVesselSemanticSegDataSet\\3DBuilderVesselRecognition_RandTranslation'

#pascal_berkeley_root = 'D:\\VOCtrainval_11-May-2012\\benchmark' #empty benchmark folder

pascal_berkeley_root = 'D:\\TryTensorflowFCN\\3DBuilderVesselSemanticSegDataSet\\benmark' #empty benchmark folder



from tf_image_segmentation.utils.pascal_voc import get_augmented_pascal_image_annotation_filename_pairs
from tf_image_segmentation.utils.tf_records import write_image_annotation_pairs_to_tfrecord

# Returns a list of (image, annotation) filename pairs (filename.jpg, filename.png)
overall_train_image_annotation_filename_pairs, overall_val_image_annotation_filename_pairs = \
                get_augmented_pascal_image_annotation_filename_pairs(pascal_root=pascal_root,                                                                                                                                                  
                pascal_berkeley_root=pascal_berkeley_root,
                mode=2)

# You can create your own tfrecords file by providing
# your list with (image, annotation) filename pairs here
'''                
write_image_annotation_pairs_to_tfrecord(filename_pairs=overall_val_image_annotation_filename_pairs,
                                         tfrecords_filename='pascal_augmented_val.tfrecords')

write_image_annotation_pairs_to_tfrecord(filename_pairs=overall_train_image_annotation_filename_pairs,
                                         tfrecords_filename='pascal_augmented_train.tfrecords')
'''


'''
write_image_annotation_pairs_to_tfrecord(filename_pairs=overall_val_image_annotation_filename_pairs,
                                         tfrecords_filename='3DBuilderVessel_augmented_val_withoutNoise.tfrecords')

write_image_annotation_pairs_to_tfrecord(filename_pairs=overall_train_image_annotation_filename_pairs,
                                         tfrecords_filename='3DBuilderVessel_augmented_train_withoutNoise.tfrecords')
'''

write_image_annotation_pairs_to_tfrecord(filename_pairs=overall_val_image_annotation_filename_pairs,
                                         tfrecords_filename='3DBuilderVessel_augmented_val_with_Rand_Translation.tfrecords')

write_image_annotation_pairs_to_tfrecord(filename_pairs=overall_train_image_annotation_filename_pairs,
                                         tfrecords_filename='3DBuilderVessel_augmented_train_with_Rand_Translation.tfrecords')