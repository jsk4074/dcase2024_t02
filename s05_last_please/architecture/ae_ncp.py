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

        self.encoder_1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.decoder_1 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.encoder_2 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.decoder_2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.encoder_3 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.decoder_3 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.wiring = AutoNCP(1024, 128)
        self.ncp = LTC(98304, self.wiring, batch_first=True)
        self.drop = nn.Dropout(p=0.2)

        self.fc = nn.Sequential(
            nn.Linear(128, 32),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Sigmoid(),
        )

        self.hidden_state = None

    def forward(self, x1, x2, x3):
        # Feature extraction(Autoencoder)
        e1 = self.encoder_1(x1)
        d1 = self.decoder_1(e1)

        e2 = self.encoder_2(x2)
        d2 = self.decoder_2(e2)
        
        e3 = self.encoder_3(x3)
        d3 = self.decoder_3(e3)

        # Concat model output
        x = torch.cat((e1, e2, e3), 1)
        # x = torch.cat((d1, d2, d3), 1)
        
        x = x.view(x.size(0), -1)
        # print("="* 20, "DEBUG", "="* 20)
        # print(x.size())
        # print("="* 20, "DEBUG", "="* 20)

        # LNN(LTC)
        x, hidden_states = self.ncp(x)
        x = self.drop(x)
        x = nn.ReLU()(x)

        x = self.fc(x)

        return d1, d2, d3, x



