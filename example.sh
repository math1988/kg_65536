# Download all the images references in train.json to /root/train_data
python get_images.py train.json /root/train_data_raw
python get_images.py validate.json /root/validate_data_raw

# Resize all training images to same size
# Images will be stored as files <--output_dir>/Size_<--image_size>/Class_XXX/YYY.jpeg
python pic_resize.py --input_dir=/root/train_data_raw --output_dir=/root/train_data_resized --image_size=224

# Augment the resized the images and store them in hdf5 format.
# Results will be stored as <--image_folder>/Size_<--image_size>/Augmentation_True#XXX.hdf5
# where XXX denotes the number of the times this function gets run, we put that in file name to not overwrite
# previous result, so that you can keep multiple versions of augmentations.
python gen_hdf_tf.py --image_folder=/root/train_data_resized --image_size=224 --augmentation=True # This will create /root/train_data_resized/Size_224/Augmentation_True#0.hdf5
python gen_hdf_tf.py --image_folder=/root/train_data_resized --image_size=224 --augmentation=True # This will create /root/train_data_resized/Size_224/Augmentation_True#1.hdf5
python gen_hdf_tf.py --image_folder=/root/train_data_resized --image_size=224 --augmentation=True # This will create /root/train_data_resized/Size_224/Augmentation_True#2.hdf5
python gen_hdf_tf.py --image_folder=/root/train_data_resized --image_size=224 --augmentation=True # This will create /root/train_data_resized/Size_224/Augmentation_True#3.hdf5
python gen_hdf_tf.py --image_folder=/root/train_data_resized --image_size=224 --augmentation=True # This will create /root/train_data_resized/Size_224/Augmentation_True#4.hdf5

# Similar process for validation data set
python pic_resize.py --input_dir=/root/validate_data_raw --output_dir=/root/validate_data_resized --image_size=224
python gen_hdf_tf.py --image_folder=/root/validate_data_resized --image_size=224 --augmentation=False

# Run the training.
# |base_model_name| self explained
# If |base_model_trainable| = False then we fix the base model parameter by a standard well trained machine, and not
# change them during training process, otherwise, we will change those model parameter based on training sample fitting.
# |image_size| self explained, and this determines the input size of the model.
# |train_data_folder| and |validate_data_folder| matches the image folder you passed in in previous resize and hdf
# generating command. (Do NOT include the "Size_XXX", which is handled internally.)
# |save_model_folder| determines the place for us to store trained model file. This script will create a unique
# subfolder for each different model, and the checkpoints (model with "partially" trained parameters) are stored.
python train.py --base_model_name=DenseNet169 --base_model_trainable=False \
                --top_fc_model_sizes=[500,128] \
                --image_size=224 \
                --train_data_folder=/root/train_data_resized \
                --validate_data_folder=/root/validate_data_resized \
                --save_model_folder=/root/models