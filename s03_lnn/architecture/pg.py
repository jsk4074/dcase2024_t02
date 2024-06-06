import torch
import torch.nn as nn
# import torch.optim as optim
# from torchvision import transforms
# from torch.utils.data import Dataset
# from torch.utils.data import DataLoader

import torchvision.models as models 
import torchvision.transforms as transforms

from ncps.torch import CfC
from ncps.torch import LTC
from ncps.wirings import AutoNCP

import numpy as np

torch.manual_seed(7777)
np.random.seed(7777)

# Define model 
class ncp(nn.Module):
    def __init__(self):
        super(ncp, self).__init__()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ).to(self.device)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        ).to(self.device)

        self.fe = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # LTC layer 
        # AutoNCP("neuron count", "Motor neuron / Ouput size")
        # LTC("Input size", self.wiring, batch_first=True)
        self.wiring = AutoNCP(64, 32)
        self.ncp = LTC(2048, self.wiring, batch_first=True)

        self.fc = nn.Sequential(
            # nn.Linear(2048, 512),
            # nn.ReLU(),
            # nn.Linear(512, 256),
            # nn.ReLU(),
            # nn.Linear(256, 128),
            # nn.ReLU(),
            nn.Linear(32, 2),
            nn.Sigmoid(),
        )

        self.hidden_state = None

    def forward(self, x):
        # Autoencoder
        x = self.encoder(x)
        x = self.decoder(x)

        # Feature extraction 
        x = self.fe(x)

        # Faltten layer 
        # flattened_size = x.size(1) * x.size(2) * x.size(3)
        # x = torch.flatten(x, 1)
        x = x.view(x.size(0), -1)

        # print("="*30, "DEBUG", "="*30)
        # print(x.size())
        # print("="*30, "DEBUG", "="*30)


        x, hidden_states = self.ncp(x)


        x = self.fc(x)

        return x



