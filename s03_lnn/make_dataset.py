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

# Make custom dataset 
class CustomDataset(Dataset): 
    def __init__(self, pkl_path, transform=None, domain=0, crop=False, feature_extraction=False):
        self.transform = transform
        # self.path = glob("../data/unziped/*/train/*.wav")
        # print("Found file counts :", len(self.path))

        # Loading raw data
        print("="*20, "Loading data", "="*20)
        # self.audio_data = [[librosa.load(i, sr=16e3)[0], i.split("/")[3]] for i in tqdm(self.path)]
        with open(pkl_path, 'rb') as f:
            self.raw_data = pkl.load(f)
        print(".pkl data has been loaded")
        print("Loaded data path :", pkl_path)

        # Resizeing feature 
        if crop:
            print("Resizeing image ...") 
            self.crop_sec = 4 
            self.audio_data = [np.array(i[0][:int(16e3) * self.crop_sec]) for i in tqdm(self.raw_data) if i[2] == domain] 

        # Feature extraction 
        if feature_extraction:
            print("="*20, "Extracting features", "="*20) 
            self.feature_data = [librosa.feature.mfcc(y = i, sr = 16e3, n_mfcc=128,) for i in tqdm(self.audio_data)] 

        # Load data
        self.feature_data = [[i[0]] for i in self.raw_data] 
        
        # Label 
        self.label_data = [[i[-1]] for i in self.raw_data] 
        print(self.label_data[0])

    # Data size return 
    def __len__(self): 
        self.feature_len = len(self.feature_data)
        self.label_len = len(self.label_data)

        if self.feature_len == self.label_len: return self.feature_len
        else: return print("Some thing wrong with data")

    # Index to data mapping 
    def __getitem__(self, idx): 
        x = torch.FloatTensor(self.feature_data[idx])
        resize_transform = transforms.Resize((128, 128))
        x = resize_transform(x)
        y = torch.LongTensor(self.label_data[idx])
        # y = self.label_data[idx]
        return x, y