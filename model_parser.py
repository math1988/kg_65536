from keras.applications.densenet import DenseNet169
from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization, GlobalMaxPool2D, Concatenate
from keras.models import Model
import tensorflow as tf

def parse_model(base_model_name, base_model_trainable, top_fc_model_sizes, pic_size):
    if base_model_name == "DenseNet169":
        with tf.device('/cpu:0'):
            inp = Input(shape=(pic_size, pic_size,3))
            raw_model = DenseNet169(include_top=False,weights='imagenet',input_shape = (pic_size,pic_size,3))(inp)
            fc = Flatten(input_shape=(9, 9, 1664))(raw_model)
            for size in top_fc_model_sizes:
                fc = Dense(size,kernel_initializer='ones',activation='sigmoid')(fc)
            model= Model(inputs= inp, outputs= fc)
            model.layers[1].trainable = base_model_trainable
    else:
        raise Exception("Unknown Model!")
    return model

def get_model_descriptor(base_model_name, base_model_trainable, top_fc_model_sizes, pic_size):
    return "{0}#Trainable_{1}#TopLayerSizes_{2}#ImageSize_{3}"\
        .format(base_model_name, base_model_trainable, "_".join(str(e) for e in top_fc_model_sizes), pic_size)


# Main function is debug use only
if __name__ == "__main__":
    print(len(parse_model("DenseNet169", False, [128], 224).layers))