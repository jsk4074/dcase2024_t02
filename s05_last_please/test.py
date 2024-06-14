import torch
import torch.nn as nn
import torch.optim as optim
# from torchvision import transforms
# from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# import torchvision.models as models 
# import torchvision.transforms as transforms

from torchviz import make_dot
import gc

# import librosa 
import numpy as np

from tqdm import tqdm 
from glob import glob 
from make_dataset import CustomDataset

torch.manual_seed(7777)
np.random.seed(7777)

def accuracy(predictions, labels):
    _, predicted_classes = torch.max(predictions, dim=1)  
    correct = (predicted_classes == labels).float()  
    acc = correct.sum() / len(correct) 
    return acc

def dataset_prep(path, batch_size, shuffle=True, mode="train"):
    dataset = CustomDataset(
        pkl_path = path, 
        domain=1,
        mode="test" 
    )

    train_loader = DataLoader( 
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
    ) 

    return train_loader

def model_test(batch_size, dataset_path, model, criterion = nn.CrossEntropyLoss(), threshold = 0.4):
    # Load dataset
    train_loader = dataset_prep(dataset_path, batch_size, shuffle=False, mode="test")

    # Autoencoder loss
    ae_criterion = nn.MSELoss()
    # LTC loss
    criterion = nn.CrossEntropyLoss()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    # print("============================ Test / Validation mode ============================")
    total_anomaly = 0
    total_correct = 0
    total_samples = 0

    for img1, img2, img3, label in train_loader:
        # Make prediction for loss calculation
        img1 = img1.to(device)
        img2 = img2.to(device)
        img3 = img3.to(device)
        label = label.to(device)
        label = label.squeeze(dim=-1) 

        # print("="* 20, "DEBUG", "="* 20)
        # print(img1.size())
        # print(img2.size())
        # print(img3.size())
        # print("="* 20, "DEBUG", "="* 20)

        # Fitting
        d1, d2, d3, pred = model(img1, img2, img3)

        # Loss calculation
        ae1_loss = ae_criterion(d1, img1) 
        ae2_loss = ae_criterion(d2, img2) 
        ae3_loss = ae_criterion(d3, img3) 
        lnn_loss = criterion(pred, label) 
        total_loss = ae1_loss + ae2_loss + ae3_loss + lnn_loss

        print(total_loss.item(), label)
        # Accuracy calculation
        if total_loss.item() > threshold and label == 1: total_anomaly += 1
        elif total_loss.item() < threshold and label == 0: total_anomaly += 1

        total_correct += (pred.argmax(1) == label).type(torch.float).sum().item()
        total_samples += label.size(0)

    # Accuracy calculation for batch
    acc = total_correct / total_samples
    anomaly_acc = total_anomaly / total_samples
    print("val Accuracy:", acc * 100, end="  ")
    print("anomaly_acc:", anomaly_acc * 100, end="  ")
    print("val loss:", total_loss.item(), end="  ")
    print("threshold :", threshold)
    print("="*50)
    return acc, anomaly_acc, total_loss.item(), threshold
