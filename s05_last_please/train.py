import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchviz import make_dot
import numpy as np

from tqdm import tqdm 
from glob import glob 

from make_dataset import CustomDataset
from test import model_test

import wandb

torch.manual_seed(7777)
np.random.seed(7777)

def accuracy(predictions, labels):
    _, predicted_classes = torch.max(predictions, dim=1)  
    correct = (predicted_classes == labels).float()  
    acc = correct.sum() / len(correct) 
    return acc

def dataset_prep(path, batch_size, mode="test"):
    dataset = CustomDataset(
        pkl_path = path, 
        domain = 1,
        mode = mode
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

    ae_criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()

    # Optimizer and Loss function 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("============================ Train mode ============================")
    for epoch_count in range(epoch):
        avg_cost = 0
        total_batch = len(train_loader)
        total_correct = 0
        total_samples = 0
        
        preds = []
        for img1, img2, img3, label in tqdm(train_loader):

            # Make prediction for loss calculation
            img1 = img1.to(device)
            img2 = img2.to(device)
            img3 = img3.to(device)
            label = label.to(device)
            label = label.squeeze(dim=-1)

            # Fitting model 
            d1, d2, d3, pred = model(img1, img2, img3)

            
            # Loss caculation
            ae1_loss = ae_criterion(d1, img1) 
            ae2_loss = ae_criterion(d2, img2) 
            ae3_loss = ae_criterion(d3, img3) 
            lnn_loss = criterion(pred, label) 
            
            total_loss = ae1_loss + ae2_loss + ae3_loss + lnn_loss


            # Run through optimizer
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            avg_cost += total_loss / total_batch

            # Accuracy calculation
            acc = accuracy(pred, label) 
            total_correct += (pred.argmax(1) == label).type(torch.float).sum().item()
            total_samples += label.size(0)

        # Accuracy calculation for batch
        acc = total_correct / total_samples
        print("Train Accuracy:", acc * 100, end="  ")
        print("Loss:", total_loss.item())

        with torch.no_grad():
            test_acc, test_anomaly_acc, test_loss, test_threshold, roc_auc, roc_pauc = model_test(
                batch_size = 1,
                dataset_path = dataset_path.replace("train", "test"),
                model = model,
                threshold = total_loss.item(), 
            )

        wandb.log({
            "accuracy": acc, 
            "loss": total_loss.item(), 
            "batch_size": batch_size,
            "test_anomaly": test_anomaly_acc,
            "test_loss": test_loss, 
            "test_threshold": test_threshold,
            "test_accuracy": test_acc,
            "auc": roc_auc,
            "pauc": roc_pauc,
        }) 
        
    # Saving models
    path = "./saved_model/dev/target/"
    model_name = dataset_path.split("_")[-4]
    loss_name = str(total_loss.item())
    print("Saving trained model")
    torch.save(model, path + model_name + "_" + loss_name + ".pkl")
