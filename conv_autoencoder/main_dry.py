import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchviz import make_dot
from torchvision import transforms
import torchvision.models as models 
import torchvision.transforms as transforms

import librosa 
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt 

from tqdm import tqdm 
from glob import glob 

# Custom files 
from models.autoencoder import Autoencoder
from models.autoencoder_fc import autoencoder_fc
from train_dry import model_fit

def main():

    batch_size = 1
    learning_rate = 1e-4
    epoch = 4
    dataset_path = glob("./data/features/classes/test*.pkl")[0]

    model_fit(
        batch_size,
        learning_rate,
        epoch,
        dataset_path,
        resnet(),
        mode = "train",
    )

if __name__ == "__main__": main()