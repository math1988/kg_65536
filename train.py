import numpy as np
import os
import tensorflow as tf
import pandas as pd
import PIL
import math
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
from keras.applications.densenet import DenseNet201
from keras.applications.densenet import DenseNet169
from keras.applications.densenet import DenseNet121
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.nasnet import NASNetMobile
from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet
from keras.applications.xception import Xception
from joblib import Parallel, delayed
import gc
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras import optimizers, losses, activations, models
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization, GlobalMaxPool2D, Concatenate
import copy
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils
from keras.regularizers import l2
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import multi_gpu_model
from IPython.display import clear_output
import mxnet as mx
import h5py
from keras import callbacks
PIL.Image.MAX_IMAGE_PIXELS = 1000000000
import h5py
from keras.utils.io_utils import HDF5Matrix
import pickle
from keras.applications import mobilenet
from keras.callbacks import LambdaCallback
import time
import logging
import threading
import matplotlib.pyplot as plt
import Augmentor
import imgaug as ia
from imgaug import augmenters as iaa
import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from keras.layers import GlobalAveragePooling2D
from keras import regularizers
%matplotlib inline
import subprocess
def image_feature_wise_center(x):
    data = np.float32(x)/255
    return data
X_thread = None
y_thread = None

def SendMail(title, content):
    msg = MIMEMultipart()
    msg['Subject'] = title
    msg['From'] = 'ricciflowfinance@gmail.com'
    msg['To'] = 'cuixin.math@gmail.com'

    text = MIMEText(content)
    msg.attach(text)

    s = smtplib.SMTP('smtp.gmail.com',587)
    s.starttls()
    s.ehlo()
    s.login('ricciflowfinance@gmail.com', 'IloveRicciflow!')
    s.sendmail('ricciflowfinance@gmail.com', 'ricciflowfinance@gmail.com', msg.as_string())
    s.quit()


LR = 0.01
BATCH_SIZE = 16
PIC_SHAPE = 299
DATE_STR = '0413'
MODEL_NAME = 'DenseNet_tunefc'
MODEL_EXPORT = '/home/mathematics/kaggle/furniture/model/'
TEST_HDF5_PATH = '/media/mathematics/kaggle/'
OPT = optimizers.SGD(lr=LR)
with tf.device('/cpu:0'):
    inp = Input(shape=(PIC_SHAPE, PIC_SHAPE,3))
    raw_model = DenseNet169(include_top=False,weights='imagenet',input_shape = (299,299,3))(inp)
    flat = Flatten(input_shape=(9, 9, 1664))(raw_model)
    fc128 = Dense(128,kernel_initializer='ones')(flat)
    model= Model(inputs= inp, outputs= fc128)
    model.layers[1].trainable = False

multi_model = multi_gpu_model(model, 2)
multi_model.compile(loss='categorical_crossentropy', optimizer=OPT,metrics=['accuracy'])

save_single_gpu_model = LambdaCallback(on_epoch_end = lambda epoch, logs: model.save(MODEL_EXPORT + 
                                      DATE_STR + 
                                      '/' +
                                      MODEL_NAME +
                                      '-'+
                                      str(PIC_SHAPE) +
                                      "-"+ 
                                      str(epoch)+ 
                                      '-' + 
                                      str(round(logs['val_acc'],3)) 
                                      +'.hdf5'))
email_result = LambdaCallback(on_epoch_end = lambda epoch, logs: SendMail(str(logs['val_acc']), 'Hi~'))
def step_decay(epoch):
    initial_lrate = 0.01
    drop = 0.5
    epochs_drop = 3.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate
lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate, save_single_gpu_model, email_result]
X_test = HDF5Matrix(TEST_HDF5_PATH + 'valid.hdf5', 'X', normalizer = image_feature_wise_center)
y_test = HDF5Matrix(TEST_HDF5_PATH + 'valid.hdf5', 'y')
for epoch in range(40):
    COMMAND = "python \"/home/mathematics/gen_hdf_tf.py\" \"/home/mathematics/kaggle/furniture/data/train/299/\" \"" + TEST_HDF5_PATH+ str(epoch+1) + ".hdf5\" False"
    process = subprocess.Popen([COMMAND],shell=True)
    X_train = HDF5Matrix(TEST_HDF5_PATH + str(epoch) +'.hdf5', 'X', normalizer = image_feature_wise_center)
    y_train = HDF5Matrix(TEST_HDF5_PATH + str(epoch) +'.hdf5', 'y')
    multi_model.fit(x = X_train, 
                    y = y_train, 
                    epochs = epoch + 1,
                    initial_epoch = epoch,
                    shuffle = 'batch',
                    validation_data = (X_test, y_test),
                    callbacks = callbacks_list)
    if (epoch > 0):
        os.system("rm "+ TEST_HDF5_PATH+ str(epoch) + ".hdf5")
    process.wait()
