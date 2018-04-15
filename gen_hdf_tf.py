# import packages
import sys
import time
import numpy as np
import h5py
import os
import pandas as pd
from joblib import Parallel, delayed
import PIL
import shutil
import imgaug as ia
from imgaug import augmenters as iaa




# This function is going called parallelly to sample images.
def sample_func(ind, aug):
    pic = np.uint8(np.array(PIL.Image.open(ind[0])))
# one hot label = ind[1]
    label = np.uint8(np.zeros(128))
    label[ind[1]] = 1
    if aug:
        np.random.seed()
        ia.seed(np.random.randint(2147483647, size = 1))
        seq = iaa.Sequential([
            iaa.Fliplr(0.5), # random crops
            # Small gaussian blur with random sigma between 0 and 0.5.
            # But we only blur about 50% of all images.
            iaa.Sometimes(0.5,
                iaa.GaussianBlur(sigma=(0, 0.1))
            ),
            iaa.Sometimes(0.5,
                iaa.CoarseDropout((0.0, 0.005), size_percent=(0.03, 0.05),per_channel = True)
            ),
            # Strengthen or weaken the contrast in each image.
            iaa.ContrastNormalization((0.75, 1.5)),
            # Add gaussian noise.
            # For 50% of all images, we sample the noise once per pixel.
            # For the other 50% of all images, we sample the noise per pixel AND
            # channel. This can change the color (not only brightness) of the
            # pixels.
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
            # Make some images brighter and some darker.
            # In 20% of all cases, we sample the multiplier once per channel,
            # which can end up changing the color of the images.
            iaa.Multiply((0.8, 1.2), per_channel=0.2),
            # Apply affine transformations to each image.
            # Scale/zoom them, translate/move them, rotate them and shear them.
            iaa.Affine(
                scale={"x": (1, 1.2), "y": (1, 1.2)},
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                rotate=(-10, 10),
                shear=(-6, 6),
                mode=ia.ALL, cval=(0, 255)
            ),

        ], random_order=True)
        pic = seq.augment_image(pic)
    return pic,label

if __name__ == "__main__":
    # constant that need to modified
    N_JOBS = 4
    BATCH_SIZE = 2048
    LOCK_PATH = "c:\\temp\\lock.kaggle"
    try:
        PIC_FOLDER = sys.argv[1]
        EXPORT_FILE = sys.argv[2]
        AUG_MARK = sys.argv[3]
        if AUG_MARK == 'True':
            AUG_MARK = True
        elif AUG_MARK == 'False':
            AUG_MARK = False
        else:
            raise Exception('Wrong input!')
    except:
        print('call format: python gen_hdf.py <PIC_FOLDER> <EXPORT_FILE>')



    path_label_list = []
    for subdir, dirs, files in os.walk(PIC_FOLDER):
        for file in files:
            class_label = int(subdir.split(os.sep)[-1])
            filepath = subdir + os.sep + file
            path_label_list.append((filepath,class_label))


    # get image size
    temp_pic = np.float32(np.array(PIL.Image.open(path_label_list[0][0])))
    pic_size = temp_pic.shape[0]
    temp_pic = None

    from_index = 0
    to_index = from_index + BATCH_SIZE
    MAX_INDEX = len(path_label_list)

    hdf5_file = h5py.File(EXPORT_FILE, mode='w')
    hdf5_file.create_dataset("X",
                            (0, pic_size, pic_size, 3),
                            maxshape = (None, pic_size, pic_size, 3),
                            dtype = np.uint8,
                            chunks = (BATCH_SIZE,pic_size,pic_size,3))

    hdf5_file.create_dataset("y",
                            (0,128),
                            maxshape=(None, 128),
                            dtype = np.uint8,
                            chunks=(BATCH_SIZE, 128))
    hdf5_file.close()
    np.random.shuffle(path_label_list)

    while(from_index < MAX_INDEX):
        sample_func_first_arg_list = []
        for i in range(from_index, to_index):
            sample_func_first_arg_list.append(path_label_list[i])

        res_list = Parallel(n_jobs=N_JOBS, verbose = 0)(delayed(sample_func)(first_arg, AUG_MARK) for first_arg in sample_func_first_arg_list)
        X = []
        y = []
        for res in res_list:
            X.append(res[0])
            y.append(res[1])
        X = np.uint8(X)
        #X = np.rollaxis(X, 3, 1)
        y = np.uint8(y)
        res_list = None
        batch_size = to_index - from_index

        with h5py.File(EXPORT_FILE, 'a') as f:
            f["X"].resize((f["X"].shape[0] + batch_size), axis = 0)
            f["X"][-batch_size:] = X
            f["y"].resize((f["y"].shape[0] + batch_size), axis = 0)
            f["y"][-batch_size:] = y
            f.close()
        from_index = to_index
        to_index = min(to_index + batch_size, MAX_INDEX)
   
