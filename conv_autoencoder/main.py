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
from models.resnet import resnet
from make_dataset import CustomDataset
from train import model_fit

import wandb

torch.manual_seed(7777)
np.random.seed(7777)

domain = ['source', 'target']
class_names = ['ToyTrain', 'gearbox', 'ToyCar', 'bearing', 'valve', 'fan', 'slider']

def main(config = None): 
    model_name = "2D_DUAL_AE_32_8_ALL_BN"
    with wandb.init(project = "dcase_2024_t02", name = model_name,):
        model_fit(
            wandb.config["batch_size"],
            wandb.config["learning_rate"],
            wandb.config["epoch"],
            wandb.config["dataset_path"],
            Autoencoder(),
        )


# def main():

#     batch_size = 256
#     learning_rate = 1e-4
#     epoch = 40
#     dataset_path = glob("./data/features/classes/train*.pkl")[0]

#     model_name = "2D_DUAL_AE_32_8_ALL_BN"
#     with wandb.init(project = "dcase_2024_t02", name = model_name,):
#         model_fit(
#             batch_size,
#             learning_rate,
#             epoch,
#             dataset_path,
#             Autoencoder(),
#         )

if __name__ == "__main__": 
    # main()
    wandb.login()
    sweep_configuration = {
        "method": "grid",

        "metric": {
            "goal": "maximize",
            "name": "accuracy"
        },

        "parameters": {
            "learning_rate": {
                "values": [1e-2, 1e-3, 1e-4, 1e-5]

                # "max": 1e-2, 
                # "min": 1e-5,
            },
            "batch_size": {
                "values": [1, 4, 8, 16, 32]
            },
            "epoch": {
                "values": [100]
            },
            "dataset_path": {
                "values": glob("/home/tori/Desktop/workspace/dcase2024/conv_autoencoder/data/features/classes/train*.pkl"),
            },
        },
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project="dcase_2024_t02")
    wandb.agent(sweep_id, function=main, count=50)