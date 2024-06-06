import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchsummary import summary
# from torch.utils.data import Dataset
# from torch.utils.data import DataLoader

# from torchviz import make_dot
# from torchvision import transforms
# import torchvision.models as models 
# import torchvision.transforms as transforms

import librosa 
import numpy as np
# import pickle as pkl
# import matplotlib.pyplot as plt 

# from tqdm import tqdm 
from glob import glob 

# Custom files 
# from models.autoencoder import Autoencoder
# from models.ae_cpe import ae_cpe
# from models.ae_liquid import vision_lnn
from architecture.ae_ncp import ncp
from architecture.pg import ncp
# from make_dataset import CustomDataset
# from train import model_fit
from train_dry import model_fit

import wandb

torch.manual_seed(7777)
np.random.seed(7777)

domain = ['source', 'target']
class_names = ['ToyTrain', 'gearbox', 'ToyCar', 'bearing', 'valve', 'fan', 'slider']

def main(config = None): 
    model = ncp()
    model_name = "NCP_MSE_LOSS_2D_SINGLE_AE_32_4_ALL_BN_x3"
    dataset_path = "./data/features/classes/train_sr_16e3_bearing_crop4_featuremfccADD_labelx3.pkl"
    model_fit(
        batch_size = 32,
        learning_rate = 1e-4,
        epoch = 50,
        dataset_path = dataset_path,
        model = model,
    )

if __name__ == "__main__": 
    main()