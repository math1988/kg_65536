import numpy as np
import sys, os, datetime, glob
import tensorflow as tf
import PIL
import math
from keras.models import Model
from joblib import Parallel, delayed
import gc
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
import h5py
from keras import callbacks
import h5py
from keras.utils.io_utils import HDF5Matrix
import pickle
from keras.applications import mobilenet
from keras.callbacks import LambdaCallback
import time
import logging
import threading
import matplotlib.pyplot as plt
import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from keras.layers import GlobalAveragePooling2D
from keras import regularizers
import subprocess
from flag_parser import parse_flag
from model_parser import parse_model, get_model_descriptor

PIL.Image.MAX_IMAGE_PIXELS = 1000000000

X_thread = None
y_thread = None

def image_feature_wise_center(x):
    data = np.float32(x)/255
    return data

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


def step_decay(epoch):
    initial_lrate = 0.01
    drop = 0.5
    epochs_drop = 3.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

def current_timestamp():
    return str(datetime.datetime.now()).replace(" " ,"-")

def Run():
    try:
        flag_map = parse_flag(sys.argv)
        base_model_name = flag_map["base_model_name"]

        top_fc_model_sizes_str = flag_map["top_fc_model_sizes"]
        top_fc_model_sizes_str = top_fc_model_sizes_str[1:-1].split(",")
        top_fc_model_sizes = []
        for size in top_fc_model_sizes_str:
            top_fc_model_sizes.append(int(size))

        pic_size = int(flag_map["image_size"])

        base_model_trainable = flag_map["base_model_trainable"]
        if base_model_trainable == "True":
            base_model_trainable = True
        elif base_model_trainable == "False":
            base_model_trainable = False
        else:
            raise Exception("base_model_trainable value error")

        train_data_root = flag_map["train_data_folder"]
        validate_data_root = flag_map["validate_data_folder"]
        save_model_root = flag_map["save_model_folder"]
    except:
        print("Syntax: {0} --base_model_name=<DenseNet169|...>\\\n"
              "--base_model_trainable=<True|False/>\\\n"
              "--top_fc_model_sizes=[<size array>]\\\n"
              "--image_size=<Number/>\\\n"
              "--train_data_folder=<Directory/>\\\n"
              "--validate_data_folder=<Directory/>\\\n"
              "--save_model_folder=<Directory/>\\\n", sys.argv[0])
        sys.exit(0)

    LR = 0.01
    BATCH_SIZE = 16

    OPT = optimizers.SGD(lr=LR)

    model_descriptor = get_model_descriptor(base_model_name, base_model_trainable, top_fc_model_sizes, pic_size)
    model_folder = save_model_root + os.sep + model_descriptor
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    list_of_files = glob.glob('{0}/*.hdf5'.format(model_folder))  # * means all if need specific format then *.csv
    if len(list_of_files) == 0:
        model = parse_model(base_model_name, base_model_trainable, top_fc_model_sizes, pic_size)
    else:
        latest_file = max(list_of_files, key=os.path.getctime)
        model = keras.models.load_model(latest_file)

    multi_model = multi_gpu_model(model, 2)
    multi_model.compile(loss='categorical_crossentropy', optimizer=OPT,metrics=['accuracy'])

    save_single_gpu_model = LambdaCallback(
        on_epoch_end = lambda epoch, logs: model.save("{0}/{1}#Acc{2}.hdf5".
                                                      format(model_folder,
                                                             current_timestamp(),
                                                             str(round(logs['val_acc'],3)))))

    email_result = LambdaCallback(on_epoch_end = lambda epoch, logs: SendMail(str(logs['val_acc']), 'Hi~'))


    lrate = LearningRateScheduler(step_decay)
    callbacks_list = [lrate, save_single_gpu_model, email_result]

    validate_data_folder = "{0}/Size_{1}".format(validate_data_root, pic_size)
    list_of_validation_files = glob.glob("{0}/*.hdf5".format(validate_data_folder))
    if len(list_of_validation_files)==0:
        print("No validation data set found!")
        sys.exit(0)
    X_test = HDF5Matrix(list_of_validation_files[0], 'X', normalizer = image_feature_wise_center)
    y_test = HDF5Matrix(list_of_validation_files[0], 'y')

    train_data_folder = "{0}/Size_{1}".format(train_data_root, pic_size)
    list_of_train_files = glob.glob("{0}/*.hdf5".format(train_data_folder))
    if len(list_of_train_files)==0:
        print("No training data set found!")
        sys.exit(0)
    num_of_augmentations = len(list_of_train_files)
    for epoch in range(40):
        # If you want to generate hdf5 on the fly, do it here, call into the function in gen_hdf_tf.py, do NOT start a
        # command process.
        X_train = HDF5Matrix(list_of_train_files[epoch % num_of_augmentations], 'X', normalizer = image_feature_wise_center)
        y_train = HDF5Matrix(list_of_train_files[epoch % num_of_augmentations], 'y')
        multi_model.fit(x = X_train,
                        y = y_train,
                        epochs = epoch + 1,
                        initial_epoch = epoch,
                        shuffle = 'batch',
                        validation_data = (X_test, y_test),
                        callbacks = callbacks_list)

if __name__ == "__main__":
    Run()
