# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 09:58:10 2017

@author: Ben
"""



import tensorflow as tf
import numpy as np
import skimage.io as io
import os, sys
from matplotlib import pyplot as plt

# Use second GPU -- change if you want to use a first one
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# Add a path to a custom fork of TF-Slim
# Get it from here:
# https://github.com/warmspringwinds/models/tree/fully_conv_vgg
sys.path.append("models/slim/")

# Add path to the cloned library
sys.path.append("tf-image-segmentation/")

checkpoints_dir = 'checkpoints'
log_folder = 'log_folder'


slim = tf.contrib.slim
vgg_checkpoint_path = os.path.join(checkpoints_dir, 'vgg_16.ckpt')

from tf_image_segmentation.utils.tf_records import read_tfrecord_and_decode_into_image_annotation_pair_tensors
from tf_image_segmentation.models.fcn_32s import FCN_32s, extract_vgg_16_mapping_without_fc8

from tf_image_segmentation.utils.pascal_voc import pascal_segmentation_lut

from tf_image_segmentation.utils.training import get_valid_logits_and_labels

from tf_image_segmentation.utils.augmentation import (distort_randomly_image_color,
                                                      flip_randomly_left_right_image_with_annotation,
                                                  scale_randomly_image_with_annotation_with_fixed_size_output)

epochs=300
vesselBatch_size=32 #32 #batch size has to be smaller when image size gets larger
                    #or GPU OOM will be raised.
'''
vesselBatch_size=1
'''
#numOfTrainingImage=1545
numOfTrainingImage=4105 #with Translation
numOfTrainingIteration=int(numOfTrainingImage/vesselBatch_size)
gpu_memory_fraction=0.7 #restrict the program from using GPU memory up to 70%.
image_train_size = [224, 224 ] #[384, 384]
#image_train_size = [384, 384 ] #[384, 384]
number_of_classes = 2 #because Pascal dataset has 21 classes

base_lr=0.000001 #default lr
'''
base_lr=1e-10
'''


 
#trying to train pascal dataset
'''
tfrecord_filename = 'pascal_augmented_train.tfrecords'
'''
   
#trying to train 3DBuilderVesselSemanticSeg dataset
#tfrecord_filename = '3DBuilderVessel_augmented_train_withoutNoise.tfrecords'
tfrecord_filename = '3DBuilderVessel_augmented_train_with_Rand_Translation.tfrecords'

pascal_voc_lut = pascal_segmentation_lut()
class_labels = list(pascal_voc_lut) #[0,1,2~20]

'''
filename_queue = tf.train.string_input_producer(
    [tfrecord_filename], num_epochs=10)
'''

filename_queue = tf.train.string_input_producer(
    [tfrecord_filename], num_epochs=epochs)

#read training images pairs(raw images, annoted images)
image, annotation = read_tfrecord_and_decode_into_image_annotation_pair_tensors(filename_queue)

# Randomly flip the training images
image, annotation = flip_randomly_left_right_image_with_annotation(image, annotation)



# Randomly scale the training images
resized_image, resized_annotation = scale_randomly_image_with_annotation_with_fixed_size_output(image, annotation, image_train_size)

#Removes dimensions of size 1 from the shape of a tensor. 
resized_annotation = tf.squeeze(resized_annotation)

image_batch, annotation_batch = tf.train.shuffle_batch( [resized_image, resized_annotation],
                                             batch_size=vesselBatch_size,
                                             capacity=3000,
                                             num_threads=2,
                                             min_after_dequeue=1000)

upsampled_logits_batch, vgg_16_variables_mapping = FCN_32s(image_batch_tensor=image_batch,
                                                           number_of_classes=number_of_classes,
                                                           is_training=True)


valid_labels_batch_tensor, valid_logits_batch_tensor = get_valid_logits_and_labels(annotation_batch_tensor=annotation_batch,
                                                                                     logits_batch_tensor=upsampled_logits_batch,
                                                                                    class_labels=class_labels)



cross_entropies = tf.nn.softmax_cross_entropy_with_logits(logits=valid_logits_batch_tensor,
                                                          labels=valid_labels_batch_tensor)

# Normalize the cross entropy -- the number of elements
# is different during each step due to mask out regions
cross_entropy_sum = tf.reduce_mean(cross_entropies)

pred = tf.argmax(upsampled_logits_batch, dimension=3)

probabilities = tf.nn.softmax(upsampled_logits_batch)


#use Adam Optimizer
with tf.variable_scope("adam_vars"):
         train_step = tf.train.AdamOptimizer(learning_rate=base_lr).minimize(cross_entropy_sum)


# VGG16 Variable's initialization functions
vgg_16_without_fc8_variables_mapping = extract_vgg_16_mapping_without_fc8(vgg_16_variables_mapping)


init_fn = slim.assign_from_checkpoint_fn(model_path=vgg_checkpoint_path,
                                         var_list=vgg_16_without_fc8_variables_mapping)

global_vars_init_op = tf.global_variables_initializer()

tf.summary.scalar('cross_entropy_loss', cross_entropy_sum)

merged_summary_op = tf.summary.merge_all()

summary_string_writer = tf.summary.FileWriter(log_folder)

# Create the log folder if doesn't exist yet
if not os.path.exists(log_folder):
     os.makedirs(log_folder)
    
#The op for initializing the variables.
local_vars_init_op = tf.local_variables_initializer()

combined_op = tf.group(local_vars_init_op, global_vars_init_op)

# We need this to save only model variables and omit
# optimization-related and other variables.
model_variables = slim.get_model_variables()
saver = tf.train.Saver(model_variables)

#restrict the maximum usage of gpu memory
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction


#with tf.Session()  as sess:
#Open a session for training using Adam Optimizer
with tf.Session(config=config)  as sess:

    
    sess.run(combined_op)
    init_fn(sess)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    
    # 10 epochs
    #for i in range(11127 * 10):
    
    #create cross entropy accumulation list for plot after training 
    crossEntropyAccumList=[]
    
    #save model for each n epochs
    saveModerForEachN=50
    
    for i in range(numOfTrainingIteration * epochs   ):
        
    
        cross_entropy, summary_string, _ = sess.run([ cross_entropy_sum,
                                                      merged_summary_op,
                                                      train_step ])
        
        print("step :" + str(i) +" Current loss: " + str(cross_entropy))
        
        
        
        
        
        summary_string_writer.add_summary(summary_string, i)
        
        #record cross entropy loss  when finishing each epoch
        if i % numOfTrainingIteration == 0: 
            #append cross entropy of each epoch to the accumulation list for plot after training
            crossEntropyAccumList.append(cross_entropy)
        
        #save model when finishing each 100 epochs
        if i % (numOfTrainingIteration*saveModerForEachN) == 0:
            save_path = saver.save(sess, "./3DBuilderVesselModelForFCN/FCN32_Model/300Epochs/model_fcn32s_3DVessel_300Epochs_with_Rand_TranslationData_epochNo"+str(i/(numOfTrainingIteration*saveModerForEachN)+1)+".ckpt")
            print("Model saved in file: %s" % save_path)
        
    coord.request_stop()
    coord.join(threads)
    
    #save model when training is over.
    save_path = saver.save(sess, "./3DBuilderVesselModelForFCN/FCN32_Model/300Epochs/model_fcn32s_3DVessel_300Epochs_with_Rand_TranslationData.ckpt")
    print("Model saved in file: %s" % save_path)
    
summary_string_writer.close()