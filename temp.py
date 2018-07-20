

from __future__ import division

import os
import sys
import tensorflow as tf
import skimage.io as io
import numpy as np
# Add path to the cloned library
sys.path.append("tf-image-segmentation/")

#slim
sys.path.append("models/slim/")

#fcn_16s_checkpoint_path = 'TrainedModel\fcn_8s\fcn_8s_checkpoint\model_fcn8s_final.ckpt'



#os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

slim = tf.contrib.slim

from tf_image_segmentation.models.fcn_8s import FCN_8s

from tf_image_segmentation.utils.inference import adapt_network_for_any_size_input
from tf_image_segmentation.utils.pascal_voc import pascal_segmentation_lut

#number_of_classes = 21
number_of_classes = 2
#currently best model
#fcn_8s_checkpoint_path='./3DBuilderVesselModelForFCN/FCN8_Model/model_fcn8s_3DVessel_30Epochs_withoutALLFC_3Classes.ckpt'

#randomly translate training image
fcn_8s_checkpoint_path='./3DBuilderVesselModelForFCN/FCN8_Model/300Epochs/model_fcn8s_3DVessel_300Epochs_with_Rand_TranslationData.ckpt'


#image_filename = '3DBuilderVesselSemanticSegDataSet/3DBuilderVesselRecognition/JPEGImages/croppedImage_a_vm_2166_1.jpg'
#image_filename = 'vessel/infatvessel2.jpg'
image_filename='vessel/croppedImage_a_vm_2210_0.jpg'
#image_filename = 'me.jpg'

image_filename_placeholder = tf.placeholder(tf.string)

feed_dict_to_use = {image_filename_placeholder: image_filename}

image_tensor = tf.read_file(image_filename_placeholder)

image_tensor = tf.image.decode_jpeg(image_tensor, channels=3)

# Fake batch for image and annotation by adding
# leading empty axis.
image_batch_tensor = tf.expand_dims(image_tensor, axis=0)

# Be careful: after adaptation, network returns final labels
# and not logits
FCN_8s = adapt_network_for_any_size_input(FCN_8s, 32)


pred, fcn_16s_variables_mapping = FCN_8s(image_batch_tensor=image_batch_tensor,
                                          number_of_classes=number_of_classes,
                                          is_training=False)

# The op for initializing the variables.
initializer = tf.local_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    
    sess.run(initializer)

    #saver.restore(sess, "TrainedModel/fcn_8s/fcn_8s_checkpoint/model_fcn8s_final.ckpt")
    
    #saver.restore(sess,  "./3DBuilderVesselModelForFCN/model_fcn8s_3DVessel.ckpt")
   
    #based on Pascal trained FCN16
    #saver.restore(sess,  "./3DBuilderVesselModelForFCN/model_fcn8s_3DVessel_BasedOnPascalFCN16.ckpt")
    
    saver.restore(sess, fcn_8s_checkpoint_path)
    
    
    
    image_np, pred_np = sess.run([image_tensor, pred], feed_dict=feed_dict_to_use)
    
    io.imshow(image_np)
    io.show()
    
    io.imshow(pred_np.squeeze())
    io.show()





# Eroding countour

import skimage.morphology

#specify the class name of the target whose contour is going to be cropped
#crop contour of vessel
targetClassName='vessel'


pascal_voc_lut = pascal_segmentation_lut()

#reverse the key(class label) and value(class name) in dict pascal_voc_lut
inv_pascal_voc_lut = {v: k for k, v in pascal_voc_lut.items()}

#get the class label of the target object
targetClassLabel=inv_pascal_voc_lut[targetClassName]

#use the class label of the target object as the mask to crop the FCN recognized target out
prediction_mask = (pred_np.squeeze() == targetClassLabel)

# Let's apply some morphological operations to
# create the contour for our sticker

cropped_object = image_np * np.dstack((prediction_mask,) * 3)

square = skimage.morphology.square(5)

temp = skimage.morphology.binary_erosion(prediction_mask, square)

negative_mask = (temp != True)

eroding_countour = negative_mask * prediction_mask

eroding_countour_img = np.dstack((eroding_countour, ) * 3)

cropped_object[eroding_countour_img] = 255 #gray value:0~255 black=> white

png_transparancy_mask = np.uint8(prediction_mask * 255)

image_shape = cropped_object.shape

png_array = np.zeros(shape=[image_shape[0], image_shape[1], 4], dtype=np.uint8)

png_array[:, :, :3] = cropped_object

png_array[:, :, 3] = png_transparancy_mask

io.imshow(cropped_object)

io.imsave('vessel/sticker_vessel.png', png_array)