import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from torchviz import make_dot


# import librosa 
import numpy as np 
from glob import glob 
from tqdm import tqdm 
import matplotlib.pyplot as plt 

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

    criterion = nn.CrossEntropyLoss()
    ae_criterion = nn.MSELoss()

    # model = torch.load("./saved_model/test_save.pkl")

    # Auto config model to fit w/ gpu or cpu 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # print("Running on :", device)

    model.to(device)
    # print("============================ Test / Validation mode ============================")
    total_anomaly = 0
    total_correct = 0
    total_samples = 0
    model_prediction = []
    label_save = []
    for img1, img2, img3, label in train_loader:
        # Make prediction for loss calculation
        img1 = img1.to(device)
        img2 = img2.to(device)
        img3 = img3.to(device)

        label = label.to(device)
        label = label.squeeze(dim=-1) 
        label_save.append(label.item())

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

        # total_loss_round = float(format(total_loss, '.3f'))
        # print(total_loss.item(),"  Label :", label.item())
        # Accuracy calculation
        # TP
        if total_loss.item() > threshold and label == 1: 
            total_anomaly += 1
            model_prediction.append(abs(threshold - total_loss.item()) / total_loss.item())

        # FN
        elif total_loss.item() <= threshold and label == 0: 
            total_anomaly += 1
            model_prediction.append(abs(threshold - total_loss.item()) / threshold)
        
        # TN
        elif label.item() == 1 and total_loss.item() <= threshold: 
            model_prediction.append(abs(1 - (abs(threshold - total_loss.item()) / threshold)))

        # FP
        elif label.item() == 0 and total_loss.item() > threshold: 
            model_prediction.append(abs(1 - abs(threshold - total_loss.item()) / total_loss.item()))

        total_correct += (pred.argmax(1) == label).type(torch.float).sum().item()
        total_samples += label.size(0)

    # Accuracy calculation for batch
    acc = total_correct / total_samples
    anomaly_acc = total_anomaly / total_samples

    fpr, tpr, thresholds = roc_curve(label_save, model_prediction)

    start_idx = np.argmax(fpr > 0.0)
    end_idx = np.argmax(fpr > 0.4)

    roc_auc = roc_auc_score(label_save, model_prediction)
    try:
        roc_pauc = roc_auc_score(label_save[start_idx:end_idx], model_prediction[start_idx:end_idx])
    except:
        roc_pauc = 0

    # print(model_prediction)

    # plt.plot(fpr, tpr)
    # plt.xlabel('FP Rate')
    # plt.ylabel('TP Rate')
    # plt.savefig("./figs/" + str(threshold) + ".png")

    print("val Accuracy:", acc * 100, end="  ")
    print("roc_auc:", roc_auc, end="  ")
    print("roc_pauc:", roc_pauc, end="  ")
    print("anomaly_acc:", anomaly_acc * 100, end="  ")
    print("val loss:", total_loss.item(), end="  ")
    print("threshold :", threshold)
    # print("thresholds :", thresholds)
    print("="*50)

    return acc, anomaly_acc, total_loss.item(), threshold, roc_auc, roc_pauc

