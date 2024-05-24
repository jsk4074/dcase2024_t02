import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torchvision.models as models 
import torchvision.transforms as transforms

from torchviz import make_dot

import librosa 
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt 

from tqdm import tqdm 
from glob import glob 

# from models.autoencoder import Autoencoder, Autoencoder_FC
from make_dataset import CustomDataset

def model_fit(batch_size, learning_rate, epoch, dataset_path, model, mode = "train", criterion = nn.CrossEntropyLoss()):
    dataset = CustomDataset(
        # pkl_path = "./data/raw_samples/train_sr_16e3.pkl", 
        # pkl_path = "./data/features/classes/train_sr_16e3_bearing_crop4_featuremfcc.pkl", 
        pkl_path = dataset_path, 
    )

    # print("=" * 20, "DEBUG", "=" * 20)
    # print(dataset.__getitem__(0))
    # print("=" * 20, "DEBUG", "=" * 20)

    train_loader = DataLoader( 
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
    ) 

    # epoch = 100
    # model = Autoencoder()

    # Auto config model to fit w/ gpu or cpu 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Training on :", device)
    model.to(device)

    # Model fit
    print("Start training ...")
    print("="*50)
    v_acc = []
    v_loss = []

    # Optimizer and Loss function 
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if mode == "test":
        with torch.no_grad():
            for epoch_count in range(epoch):
                avg_cost = 0
                total_batch = len(train_loader)
                    

                preds = []
                print("Epoch:", epoch_count + 1)
                for img, label in tqdm(train_loader):
                    # Make prediction for loss calculation
                    img = img.to(device)
                    pred = model(img)
                        
                    label = label(label)
                    
                    # Loss caculation
                    loss = criterion(pred, label)

                    # Run through optimizer
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    avg_cost += loss / total_batch

                    # Accuracy calculation
                    # pred = torch.argmax(pred, 1) == img
                    preds.append(pred.float().mean())

                # Accuracy calculation for batch
                preds = torch.tensor(preds)
                acc = preds.float().mean()

                # Save for later training metric visualizations
                v_acc.append(acc)
                v_loss.append(float(avg_cost))

                # Visualize the results
                print("Accuracy:", acc.item() * 100)
                # print("Loss:", str(float(avg_cost)).format(":e"))
                print("Loss:", loss.item())
                print("="*50)

    else: 
        for epoch_count in range(epoch):
            avg_cost = 0
            total_batch = len(train_loader)
                

            preds = []
            print("Epoch:", epoch_count + 1)
            for img, _ in tqdm(train_loader):
                # Make prediction for loss calculation
                img = img.to(device)
                pred = model(img)
                    
                # Loss caculation
                loss = criterion(pred, 0)

                # Run through optimizer
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_cost += loss / total_batch

                # Accuracy calculation
                # pred = torch.argmax(pred, 1) == img
                preds.append(pred.float().mean())

            # Accuracy calculation for batch
            preds = torch.tensor(preds)
            acc = preds.float().mean()

            # Save for later training metric visualizations
            v_acc.append(acc)
            v_loss.append(float(avg_cost))

            # Visualize the results
            print("Accuracy:", acc.item() * 100)
            # print("Loss:", str(float(avg_cost)).format(":e"))
            print("Loss:", loss.item())
            print("="*50)