import torch
import torch.nn as nn
import torch.nn.functional as F

from config import N_CHANNELS, BATCH_SIZE, IMG_SIZE



# following the description in the EUROSAT dataset paper
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv_block = nn.Sequential(
                    nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2), #64-32
                    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2), #32-16
                    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2) #16-8
                )
        self.fc = nn.Linear(in_features=self.calculate_in_features(), out_features=num_classes)
        
    def calculate_in_features(self):
        dummy_input = torch.randn(BATCH_SIZE, N_CHANNELS, IMG_SIZE, IMG_SIZE)  # Example input (batch_size=1, C=3, H=32, W=32)
        dummy_output = self.conv_block(dummy_input)
        flattened_size = dummy_output.numel() // dummy_output.shape[0]  # Divide by batch size
        return flattened_size 

    def forward(self, x):
        x = self.conv_block(x)  
        # print("after convolution", x.shape)
        x = x.view(x.size(0), -1)  # Flatten
        # print("after final view:", x.shape)
        x = self.fc(x)
        # print("after fc", x.shape)
        # print(x)
        return x

