
import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import tqdm.notebook as tqdm
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
import librosa.display as dsp

# Augmentation
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

# Model
from architecture.autoencoder import autoencoder_fc
from architecture.autoencoder import Denoise

tf.random.set_seed(7777)

def load_data():
    BATCH_SIZE = 256
    SHUFFLE_BUFFER_SIZE = 128
    AUTOTUNE = tf.data.AUTOTUNE
    
    pkl_path = "./data/features/classes/train_sr_16e3_bearing_crop4_featuremfccADD_label.pkl"
    with open(pkl_path, 'rb') as f: raw_data = pkl.load(f)
    
    train_data = [np.resize(i[0], (128, 128)) for i in raw_data]
    train_data = tf.expand_dims(train_data, -1)

    train_label = [i[-1] for i in raw_data]

    train_ds = tf.data.Dataset.from_tensor_slices((train_data, train_label))
    train_ds = train_ds.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

    return train_ds

def main():
    # pkl_path = "./data/features/classes/train_sr_16e3_bearing_crop4_featuremfccADD_label.pkl"
    # with open(pkl_path, 'rb') as f: raw_data = pkl.load(f)
    # train_data = [np.resize(i[0], (128, 128)) for i in raw_data]
    # train_data = tf.expand_dims(train_data, -1)
    # train_ds = train_data

    train_ds = load_data()


    model = autoencoder_fc()
    # model = Denoise()
    # model = tf.keras.applications.ResNet50(
    #     include_top=False,
    #     weights=None,
    #     input_shape=(128, 128, 1),
    #     pooling=None,
    #     classes=2,
    #     classifier_activation='softmax'
    # )

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(
        optimizer=optimizer,
        # loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=['accuracy'],
    )

    # model.summary()
    history = model.fit(
        train_ds, 
        # train_ds, 
        batch_size=990,
        shuffle=True,
        epochs=50,
    )

if __name__ == "__main__": main()