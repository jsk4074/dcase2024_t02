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
from s01.models.liquidnet import ae_cpe
from train_dry import model_fit

def main():
    batch_size = 32
    learning_rate = 1e-4
    epoch = 2
    dataset_path = glob("./data/features/classes/train*x3.pkl")[0]
    print("Training on", dataset_path.split("/")[-1])
    # model_name = "2D_DUAL_AE_32_8_ALL_BN"

    model_fit(
        batch_size,
        learning_rate,
        epoch,
        dataset_path,
        Autoencoder(),
    )

if __name__ == "__main__": main()