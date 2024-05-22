import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torchvision.models as models 
import torchvision.transforms as transforms


# Define model 
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )


    def forward(self, x):
        x1 = self.encoder(x)
        y1 = self.decoder(x1)

        x2 = self.encoder(y1)
        y2 = self.decoder(x2)

        # x3 = self.encoder(y2)
        # y3 = self.decoder(x3)

        return y2
# Define model 
class Autoencoder_FC(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
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
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )


    def forward(self, x):
        x1 = self.encoder(x)
        y1 = self.decoder(x1)

        x2 = self.encoder(y1)
        y2 = self.decoder(x2)

        # x3 = self.encoder(y2)
        # y3 = self.decoder(x3)

        return y2

###################################################################################
                                    # LEGACY #
###################################################################################

# # Define model 
# class Autoencoder(nn.Module):
#     def __init__(self):
#         super(Autoencoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),

#             nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),

#             nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#         )

#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),

#             nn.ConvTranspose2d(64, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),

#             nn.ConvTranspose2d(128, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.Sigmoid()
#         )

#         # self.concat = torch.cat()
         

#     def forward(self, x):

#         x1 = self.encoder(x)
#         y1 = self.decoder(x1)

#         # e1 = torch.cat((y1, x), 0, )
#         # e1 = torch.add(y1, x)
#         # e1 = torch.tensor(e1, requires_grad=True)
#         # print("="*20, "DEBUG", "="*20)
#         # print("y1 :", np.shape(y1))
#         # print("x :", np.shape(x))
#         # print("e1 :", np.shape(e1))
#         # print("="*20, "DEBUG", "="*20)


#         x2 = self.encoder(y1)
#         y2 = self.decoder(x2)

#         # e2 = torch.cat((y2, e1), 0, )
#         # e2 = torch.add(y2, e1)
#         # e2 = torch.tensor(e2, requires_grad=True)

#         x3 = self.encoder(y2)
#         y3 = self.decoder(x3)

#         # e3 = torch.cat((y3, e2), 0, )
#         # e3 = torch.add(y3, e2)
#         # e3 = torch.tensor(e3, requires_grad=True)

#         # x4 = self.encoder(y3)
#         # y4 = self.decoder(x4)

#         return y3


