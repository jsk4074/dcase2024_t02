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

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )

        self.wiring = AutoNCP(512, 128)
        self.ncp = LTC(128 * 128 * 3, self.wiring, batch_first=True)
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
        x1 = self.encoder(x1)
        x1 = self.decoder(x1)
        x1 = self.encoder(x1)
        x1 = self.decoder(x1)

        x2 = self.encoder(x2)
        x2 = self.decoder(x2)
        x2 = self.encoder(x2)
        x2 = self.decoder(x2)
        
        x3 = self.encoder(x3)
        x3 = self.decoder(x3)
        x3 = self.encoder(x3)
        x3 = self.decoder(x3)

        # x = x1 + x2 + x3
        x = torch.cat((x1, x2, x3), 1)
        
        x = x.view(x.size(0), -1)

        x, hidden_states = self.ncp(x)
        x = self.drop(x)
        x = nn.ReLU()(x)

        x = self.fc(x)

        return x



