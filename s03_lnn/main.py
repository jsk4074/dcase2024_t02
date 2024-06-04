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
from models.ae_ncp import ncp
# from make_dataset import CustomDataset
from train import model_fit

import wandb

torch.manual_seed(7777)
np.random.seed(7777)

domain = ['source', 'target']
class_names = ['ToyTrain', 'gearbox', 'ToyCar', 'bearing', 'valve', 'fan', 'slider']

def main(config = None): 
    model = ncp()
    model_name = "NCP_MSE_LOSS_2D_SINGLE_AE_32_4_ALL_BN_x3"
    # summary(model, (1, 128, 128))

    with wandb.init(project = "dcase_2024_t02", name = model_name,):
        model_fit(
            wandb.config["batch_size"],
            wandb.config["learning_rate"],
            wandb.config["epoch"],
            wandb.config["dataset_path"],
            model,
        )

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
                "values": [1e-3, 1e-4, 1e-5]
            },
            "batch_size": {
                "values": [16, 32, 64, 128]
            },
            "epoch": {
                "values": [150]
            },
            "dataset_path": {
                "values": list(glob("./data/features/classes/train*x3.pkl")),
            },
        },
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project="dcase_2024_t02")
    wandb.agent(sweep_id, function=main, count=50)