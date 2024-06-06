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

from tqdm import tqdm 
from glob import glob 
from make_dataset import CustomDataset

import wandb

torch.manual_seed(7777)
np.random.seed(7777)

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
            

        preds = []
        print("Epoch:", epoch_count + 1)
        for img, label in tqdm(train_loader):

            # Make prediction for loss calculation
            img = img.to(device)
            label = label.to(device)
            label = torch.argmax(label, dim=1) 
            pred = model(img)
            
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
        correct_prediction = torch.argmax(pred, 1) == label
        # acc = correct_prediction / total_batch
        # preds = torch.tensor(preds)
        acc = preds.float().mean()

        # Save for later training metric visualizations
        v_acc.append(acc)
        v_loss.append(float(avg_cost))

        # Visualize the results
        print("Accuracy:", acc * 100)
        # print("Accuracy:", acc.item() * 100)
        # print("Loss:", str(float(avg_cost)).format(":e"))
        print("Loss:", loss.item())
        print("="*50)
    
    path = "./saved_model/"
    model_name = ""