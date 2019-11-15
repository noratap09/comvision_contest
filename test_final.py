from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, UpSampling2D

from keras.models import load_model
import cv2
import numpy as np
#edit with your model
IMAGE_SIZE = (480,480)

from keras_unet.models import custom_unet

model = custom_unet(
    input_shape=(480, 480, 3),
    use_batch_norm=True,
    num_classes=1,
    filters=16,
    dropout=0,
    num_layers=4,
    output_activation='sigmoid')

model.load_weights('my_model_final_in_480_f_16.h5')

import glob

path = glob.glob("Test/Input/*.jpg")

for myfile in path:
    test_im = cv2.imread(myfile)
    true_size = test_im.shape
    imshow_size = (512,round(true_size[0]*512/true_size[1]))
    #cv2.imshow('Input',cv2.resize(test_im, imshow_size))

    test_im = cv2.cvtColor(test_im, cv2.COLOR_BGR2RGB)
    test_im = cv2.resize(test_im, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
    test_im = test_im/255.
    test_im = np.expand_dims(test_im, axis=0)
    segmented = model.predict(test_im)
    segmented = np.around(segmented)
    segmented = (segmented[0, :, :, 0]*255).astype('uint8')
    im_pred = cv2.resize(segmented, imshow_size)
    #cv2.imshow('Output',im_pred)
    im_pred = cv2.resize(im_pred, (true_size[1],true_size[0]), interpolation = cv2.INTER_AREA)
    #im_true =  cv2.resize(im_true, IMAGE_SIZE)
    #im_pred =  cv2.resize(im_pred, IMAGE_SIZE)
    myfile = myfile.replace("Input","Output")
    cv2.imwrite(myfile,im_pred)
    print(myfile)
    #cv2.waitKey()
