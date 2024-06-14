import torch

import numpy as np
from glob import glob 

from architecture.ae_ncp import ncp
from train import model_fit

import wandb

torch.manual_seed(7777)
np.random.seed(7777)

domain = ['source', 'target']
class_names = ['ToyTrain', 'gearbox', 'ToyCar', 'bearing', 'valve', 'fan', 'slider']

def main(config = None): 
    model = ncp()
    run =  wandb.init(config=config)
    run.name= wandb.config["dataset_path"].split("/")[-1].split("_")[3] + "_" + str(wandb.config["learning_rate"]) + "_" + str(wandb.config["batch_size"])

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
        model,
    )

    # with torch.no_grad():
    #     model_test(
    #         batch_size = 1000,
    #         dataset_path = test_path,
    #         model = model,
    #     )


if __name__ == "__main__": 
    # main()
    wandb.login()
    sweep_configuration = {
        "name": "NCP_RTX3090TI_FINAL",
        "method": "grid",
        "metric": {
            "goal": "maximize",
            "name": "accuracy"
        },
        "parameters": {
            "learning_rate": {
                "values": [1e-4]
            },
            "batch_size": {
                "values": [8]
            },
            "epoch": {
                "values": [100]
            },
            "dataset_path": {
                "values": list(glob("./data/features/stft/train*s04.pkl")),
            },
            # "test_dataset_path": {
            #     "values": list(glob("./data/features/classes/test*x3.pkl")),
            # },
        },
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project="dcase_2024_t02")
    wandb.agent(sweep_id, function=main, count=100)