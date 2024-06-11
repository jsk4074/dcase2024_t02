import torch
import torch.nn as nn
import torch.optim as optim
# from torchvision import transforms
# from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# import torchvision.models as models 
# import torchvision.transforms as transforms

from torchviz import make_dot

# import librosa 
import numpy as np
# import pickle as pkl
# import matplotlib.pyplot as plt 

from tqdm import tqdm 
from glob import glob 
from make_dataset import CustomDataset

import wandb

torch.manual_seed(7777)
np.random.seed(7777)

def accuracy(predictions, labels):
    _, predicted_classes = torch.max(predictions, dim=1)  
    correct = (predicted_classes == labels).float()  
    acc = correct.sum() / len(correct) 
    return acc

def dataset_prep(path, batch_size, mode="train"):
    dataset = CustomDataset(
        pkl_path = path, 
        domain=1,
        mode=mode
    )

    train_loader = DataLoader( 
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
    ) 

    return train_loader

def model_fit(batch_size, learning_rate, epoch, dataset_path, model, mode = "train", criterion = nn.CrossEntropyLoss()):
    dataset = CustomDataset(
        pkl_path = dataset_path, 
    )

    train_loader = DataLoader( 
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
    ) 

    # Auto config model to fit w/ gpu or cpu 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Training on :", device)
    model.to(device)

    # Model fit
    print("Start training ...")
    print("="*50)
    v_acc = []
    v_loss = []

    # Optimizer and Loss function 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("============================ Train mode ============================")
    for epoch_count in range(epoch):
        avg_cost = 0
        total_batch = len(train_loader)
        total_correct = 0
        total_samples = 0
        
        preds = []
        print("Epoch:", epoch_count + 1)
        for img, label in tqdm(train_loader):

            # Make prediction for loss calculation
            img = img.to(device)
            label = label.to(device)
            label = label.squeeze(dim=-1)

            # Fitting model 
            pred = model(img)
            
            # Loss caculation
            loss = criterion(pred, label)

            # Run through optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_cost += loss / total_batch

            # Accuracy calculation
            acc = accuracy(pred, label) 
            avg_cost += loss / total_batch
            total_correct += (pred.argmax(1) == label).type(torch.float).sum().item()
            total_samples += label.size(0)

        # Accuracy calculation for batch
        acc = total_correct / total_samples
        print("Accuracy:", acc * 100)
        print("Loss:", loss.item())
        print("="*50)

        # Visualize the results
        print("Accuracy:", acc * 100)
        # print("Loss:", str(float(avg_cost)).format(":e"))
        print("Loss:", loss.item())
        print("="*50)
            
        wandb.log({
            "accuracy": acc, 
            "loss": loss.item(), 
            "batch_size": batch_size,
        }) 