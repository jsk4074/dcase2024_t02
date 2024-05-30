
import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt

# managing files
import pickle as pkl
from glob import glob
from shutil import copyfile

# tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D, Conv2D, MaxPool2D, ZeroPadding2D, BatchNormalization, Input, DepthwiseConv2D, Add, LeakyReLU, ReLU
from tensorflow.keras.optimizers import Adam, SGD

# Audio
import librosa
# import librosa.display as dsp

# Augmentation
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

# Model
from architecture.autoencoder import autoencoder_fc
from architecture.autoencoder import autoencoder

tf.random.set_seed(7777)

BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 4
AUTOTUNE = tf.data.AUTOTUNE

def load_data(is_ae = False):
    
    pkl_path = "./data/features/classes/train_sr_16e3_bearing_crop4_featuremfccADD_labelx3.pkl"
    with open(pkl_path, 'rb') as f: raw_data = pkl.load(f)
    
    train_data = [np.resize(i[0], (128, 128)) for i in raw_data]
    train_data = tf.expand_dims(train_data, -1)

    train_label = [i[-1] for i in raw_data]

    if is_ae: 
        print("Running autoencoder mode!")
        train_ds = tf.data.Dataset.from_tensor_slices((train_data, train_data))
    else: 
        print("Running non-autoencoder mode!")
        train_ds = tf.data.Dataset.from_tensor_slices((train_data, train_label))

    train_ds = train_ds.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

    return train_ds

def main():
    ###########################################
    #              Load Datasets              # 
    ###########################################
    train_ds = load_data(is_ae = False)


    ###########################################
    #          Load And Compile Model         # 
    ###########################################
    model = autoencoder_fc()
    # loss = tf.keras.losses.BinaryCrossentropy()
    loss = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    # model = autoencoder()
    # model = tf.keras.applications.ResNet50(
    #     include_top=False,
    #     weights=None,
    #     input_shape=(128, 128, 1),
    #     pooling=None,
    #     classes=2,
    #     classifier_activation='softmax'
    # )
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy'],
        # metrics=['mean_absolute_error'],
    )

    # model.build((128, 128, 1))
    # model.summary()

    ###########################################
    #              Model fitting              # 
    ###########################################
    with tf.device("/GPU:0"):
        history = model.fit(
            train_ds, 
            batch_size=BATCH_SIZE,
            shuffle=True,
            epochs=5,
        )

if __name__ == "__main__": main()