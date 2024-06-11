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

import wandb

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

def model_test(batch_size, dataset_path, model, criterion = nn.CrossEntropyLoss()):
    # Load dataset
    train_loader = dataset_prep(dataset_path, batch_size, shuffle=False, mode="test")

    model = torch.load("./saved_model/test_save.pkl")

    # Auto config model to fit w/ gpu or cpu 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Running on :", device)

    model.to(device)
    print("============================ Test mode ============================")
    # for epoch_count in range(epoch):
    avg_cost = 0
    total_batch = len(train_loader)
    total_correct = 0
    total_samples = 0

    for img, label in tqdm(train_loader):

        # Make prediction for loss calculation
        img = img.to(device)
        label = label.to(device)
        label = label.squeeze(dim=-1) 

        # Fitting
        pred = model(img) 

        # Loss caculation
        test_loss = criterion(pred, label) 

        # Accuracy calculation
        test_acc = accuracy(pred, label) 
        avg_cost += test_loss / total_batch
        total_correct += (pred.argmax(1) == label).type(torch.float).sum().item()
        total_samples += label.size(0)

        for i in range(len(pred)):
            print(pred[i], end="")
            print(label[i])

    # Accuracy calculation for batch
    test_acc = total_correct / total_samples
    print("Accuracy:", test_acc * 100)
    print("Loss:", test_loss.item())
    print("="*50)

    wandb.log({
        "test_accuracy": test_acc, 
        "test_loss": test_loss.item(), 
        "test_batch_size": batch_size,
    })
