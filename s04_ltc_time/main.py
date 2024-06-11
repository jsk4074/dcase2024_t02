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
# from architecture.autoencoder import Autoencoder
from architecture.ae_cpe import ae_cpe
# from architecture.ae_liquid import vision_lnn
from architecture.ae_ncp import ncp
# from make_dataset import CustomDataset
from train import model_fit
from test import model_test

import wandb

torch.manual_seed(7777)
np.random.seed(7777)

domain = ['source', 'target']
class_names = ['ToyTrain', 'gearbox', 'ToyCar', 'bearing', 'valve', 'fan', 'slider']

def main(config = None): 
    model = ncp()
    model_name = "NCP_CE_LOSS_2D_QUAD_AE_32_4_ALL_BN_x3"
    run =  wandb.init(config=config)
    run.name= wandb.config["dataset_path"].split("/")[-1].split("_")[3] + "_" + str(wandb.config["learning_rate"]) + "_" + str(wandb.config["batch_size"])

    # with wandb.init(project = "dcase_2024_t02", name = "runs",):
    # with run:
    train_path = wandb.config["dataset_path"]
    test_path = train_path.replace("train", "test")
    run_name = train_path.split("/")[-1].split("_")[3]

    wandb.log({"run_name": run_name,})

    print("="*20, "DEBUG", "="*20)
    print(train_path)
    print(test_path)
    print("="*20, "DEBUG", "="*20)

    model_fit(
        wandb.config["batch_size"],
        wandb.config["learning_rate"],
        wandb.config["epoch"],
        train_path,
        # wandb.config["train_dataset_path"],
        model,
    )

    with torch.no_grad():
        model_test(
            batch_size = 1000,
            dataset_path = test_path,
            model = model,
        )


if __name__ == "__main__": 
    # main()
    wandb.login()
    sweep_configuration = {
        "name": "NCP_CE_LOSS_2D_QUAD_AE",
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
                "values": [10]
            },
            "dataset_path": {
                "values": list(glob("./data/features/classes/train*x3.pkl")),
            },
            # "test_dataset_path": {
            #     "values": list(glob("./data/features/classes/test*x3.pkl")),
            # },
        },
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project="dcase_2024_t02")
    wandb.agent(sweep_id, function=main, count=50)