import os
import zipfile
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from i3d_inception import Inception_Inflated3d



def i3d_v2():
    inputs = tf.keras.layers.Input(shape=[24, 256, 256,1])

    inputs_ = tf.keras.layers.Concatenate(axis=4)([inputs, inputs, inputs])

    NUM_FRAMES=24
    FRAME_HEIGHT=256
    FRAME_WIDTH=256

    NUM_RGB_CHANNELS=3
    base_model = Inception_Inflated3d(
        include_top=False,
        weights='rgb_imagenet_and_kinetics',
        input_shape=(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, NUM_RGB_CHANNELS),
        classes=3)
    base_model.trainable = False
    hidden = base_model(inputs_)

    #(None, 5,1,1,1024)
    hidden = tf.keras.layers.AveragePooling3D((2, 1, 1), strides=(1, 1, 1), padding='valid', name='global_avg_pool_ed2')(hidden)
    hidden = tf.keras.layers.Flatten()(hidden)

    
    hidden = tf.keras.layers.BatchNormalization()(hidden)
    hidden = tf.keras.layers.Dropout(0.5)(hidden)
    hidden = tf.keras.layers.Activation('relu')(hidden)
    hidden = tf.keras.layers.Dense(32)(hidden)
    hidden = tf.keras.layers.BatchNormalization()(hidden)
    hidden = tf.keras.layers.Dropout(0.5)(hidden)
    hidden = tf.keras.layers.Activation('relu')(hidden)
    
    hidden = tf.keras.layers.Dense(21, name="sites")(hidden)
    
    hidden = tf.keras.layers.Activation('softmax')(hidden)


    model=tf.keras.Model(inputs = inputs, outputs = hidden)
    return model


def rblock(x_, num_filters, factor):
    y = layers.Conv3D(filters=num_filters//factor, kernel_size=3,padding='same', use_bias=False, activation="relu")(x_)

    x = layers.Conv3D(filters=num_filters//factor, kernel_size=3,padding='same', use_bias=False, activation="relu")(y)
    x = layers.Conv3D(filters=num_filters//factor, kernel_size=3,padding='same', use_bias=False, activation="relu")(x)
    x = x + y
    return x


def CNN3D(depth=64, factor=4):
    """Build a 3D convolutional neural network model."""
    inputs = tf.keras.layers.Input(shape=[24, 256, 256,1])
    #inputs_ = tf.keras.layers.Concatenate(axis=4)([inputs, inputs, inputs])
    inputs_ = tf.keras.layers.Permute(dims=(2, 3, 1, 4))(inputs)
   # inputs = keras.Input((width, height, depth, 1))

    x = layers.Conv3D(filters=64//factor, kernel_size=3,padding='same', activation="relu")(inputs_)
    x = layers.MaxPool3D(pool_size=(2, 2, 1))(x)
    x = layers.BatchNormalization()(x)

    x = rblock(x, 64, factor)
    x = layers.MaxPool3D(pool_size=(2, 2, 1))(x)
    x = layers.BatchNormalization()(x)

    x = rblock(x, 128, factor)
#    x = layers.Conv3D(filters=128//factor, kernel_size=3,padding='same', activation="relu")(x)
    x = layers.MaxPool3D(pool_size=(2, 2, 2))(x)
    x = layers.BatchNormalization()(x)

    x = rblock(x, 256, factor)
 #   x = layers.Conv3D(filters=256//factor, kernel_size=3,padding='same', activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512//factor, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
#    outputs = layers.Dense(units=1, activation="sigmoid")(x)
#    hidden = tf.keras.layers.AveragePooling3D((2, 1, 1), strides=(1, 1, 1), padding='valid', name='global_avg_pool_ed2')(hidden)
    #hidden = tf.keras.layers.Flatten()(hidden)
    hidden = layers.Dense(3)(x)
    outputs = tf.keras.layers.Activation('softmax', dtype='float32')(hidden)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model
#FLAGS = flags.FLAGS
def cap_out():
     inputs = tf.keras.layers.Input(shape=[1280])
     hidden = tf.keras.layers.Flatten()(inputs)
     hidden = tf.keras.layers.BatchNormalization()(hidden)
     hidden = tf.keras.layers.Dropout(GLOBAL_DROPOUT)(hidden)
     hidden = tf.keras.layers.Dense(64)(hidden)
     cap=tf.keras.Model(inputs = inputs, outputs = hidden)
     return cap
GLOBAL_DROPOUT = 0.5

def define_r18():
     inputs = tf.keras.layers.Input(shape=[224,224*8, 3])
     base_model = tf.keras.applications.MobileNetV2(weights= 'imagenet',
                                                 include_top=False,
                                                 input_shape= (224,224,3))
     print(base_model.summary())
     base_model.trainable = False
     inputs_frame0 = tf.keras.layers.Lambda(lambda x: x[:,:,0:224,:])(inputs)
     inputs_frame1 = tf.keras.layers.Lambda(lambda x: x[:,:,224:224*2,:])(inputs)
     inputs_frame2 = tf.keras.layers.Lambda(lambda x: x[:,:,224*2:224*3,:])(inputs)
     inputs_frame3 = tf.keras.layers.Lambda(lambda x: x[:,:,224*3:224*4,:])(inputs)
     inputs_frame4 = tf.keras.layers.Lambda(lambda x: x[:,:,224*4:224*5,:])(inputs)
     inputs_frame5 = tf.keras.layers.Lambda(lambda x: x[:,:,224*5:224*6,:])(inputs)
     inputs_frame6 = tf.keras.layers.Lambda(lambda x: x[:,:,224*6:224*7,:])(inputs)
     inputs_frame7 = tf.keras.layers.Lambda(lambda x: x[:,:,224*7:224*8,:])(inputs)

     feature_batch0 = base_model(inputs_frame0)
     x0 = tf.keras.layers.GlobalAveragePooling2D()(feature_batch0)
     feature_batch1 = base_model(inputs_frame1)
     x1 = tf.keras.layers.GlobalAveragePooling2D()(feature_batch1)
     feature_batch2 = base_model(inputs_frame2)
     x2 = tf.keras.layers.GlobalAveragePooling2D()(feature_batch2)
     feature_batch3 = base_model(inputs_frame3)
     x3 = tf.keras.layers.GlobalAveragePooling2D()(feature_batch3)
     feature_batch4 = base_model(inputs_frame4)
     x4 = tf.keras.layers.GlobalAveragePooling2D()(feature_batch4)
     feature_batch5 = base_model(inputs_frame5)
     x5 = tf.keras.layers.GlobalAveragePooling2D()(feature_batch5)
     feature_batch6 = base_model(inputs_frame6)
     x6 = tf.keras.layers.GlobalAveragePooling2D()(feature_batch6)
     feature_batch7 = base_model(inputs_frame7)
     x7 = tf.keras.layers.GlobalAveragePooling2D()(feature_batch7)
     cap = cap_out()
     x0 = cap(x0)
     x1 = cap(x1)
     x2 = cap(x2)
     x3 = cap(x3)
     x4 = cap(x4)
     x5 = cap(x5)
     x6 = cap(x6)
     x7 = cap(x7)
     combined = tf.keras.layers.Concatenate(axis=1)([x0, x1,x2,x3,x4,x5,x6,x7])
     hidden = tf.keras.layers.BatchNormalization()(combined)
     hidden = tf.keras.layers.Dropout(0.5)(hidden)
     hidden = tf.keras.layers.Activation('relu')(hidden)
     hidden = tf.keras.layers.Dense(32)(hidden)
     hidden = tf.keras.layers.BatchNormalization()(hidden)
     hidden = tf.keras.layers.Dropout(0.5)(hidden)
     hidden = tf.keras.layers.Activation('relu')(hidden)
     hidden = tf.keras.layers.Dense(3)(hidden)
#     hidden = tf.keras.layers.Activation('softmax')(hidden)
     model=tf.keras.Model(inputs = inputs, outputs = hidden)
     base_learning_rate = 0.001
     initial_learning_rate = base_learning_rate
     return model



# Build model.
#model = get_model(width=128, height=128, depth=64)
#model.summary()
