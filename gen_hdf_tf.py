# import packages
import sys
import time
import numpy as np
import h5py
import os
from joblib import Parallel, delayed
import PIL
import shutil
import imgaug as ia
from imgaug import augmenters as iaa


from flag_parser import parse_flag


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

def save_jpegs_into_hdf5(pic_root_folder, pic_size, aug_mark):
    # constant that need to modified
    N_JOBS = 4
    BATCH_SIZE = 2048

    path_label_list = []
    pic_folder = "{0}/Size_{1}".format(pic_root_folder, pic_size)

    index = 0
    while True:
        export_file = "{0}/Augmentation_{1}#{2}.hdf5".format(pic_folder, aug_mark, index)
        if not os.path.exists(export_file):
            break;
        index = index + 1
    for subdir, dirs, files in os.walk(pic_folder):
        for file in files:
            if (file[-5:]!=".jpeg"):
                continue
            # Format: <Directory>/Size_<size>/Class_<label_id>/...
            class_label = int(subdir.split(os.sep)[-1].split("_")[-1])
            filepath = subdir + os.sep + file
            path_label_list.append((filepath, class_label))

    from_index = 0
    MAX_INDEX = len(path_label_list)
    to_index = min(from_index + BATCH_SIZE, MAX_INDEX)

    hdf5_file = h5py.File(export_file, mode='w')
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

        res_list = Parallel(n_jobs=N_JOBS, verbose = 0)(delayed(sample_func)(first_arg, aug_mark) for first_arg in sample_func_first_arg_list)
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

        with h5py.File(export_file, 'a') as f:
            f["X"].resize((f["X"].shape[0] + batch_size), axis = 0)
            f["X"][-batch_size:] = X
            f["y"].resize((f["y"].shape[0] + batch_size), axis = 0)
            f["y"][-batch_size:] = y
            f.close()
        from_index = to_index
        to_index = min(to_index + batch_size, MAX_INDEX)
    return export_file

def Run():
    #LOCK_PATH = "c:\\temp\\lock.kaggle"
    try:
        flag_map = parse_flag(sys.argv)

        pic_root_folder = flag_map["image_folder"]
        # Let the end user to pass in the |image_size|, in case multiple sizes data exist in the root folder.
        pic_size = int(flag_map["image_size"])
        aug_mark = flag_map["augmentation"]
        if aug_mark == 'True':
            aug_mark = True
        elif aug_mark == 'False':
            aug_mark = False
        else:
            raise Exception('Wrong input!')
    except:
        print('Syntax: {0}\\\n --image_folder=<Directory/>\\\n --image_size=<Number/>\\\n '
              '--augmentation=<True|False>'.format(sys.argv[0]))
        sys.exit(0)


    export_file_name = save_jpegs_into_hdf5(pic_root_folder,pic_size, aug_mark)
    print("Saved result to {0}".format(export_file_name))

if __name__ == "__main__":
    Run()
   
