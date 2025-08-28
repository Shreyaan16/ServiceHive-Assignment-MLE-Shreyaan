import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, num_filters=32):
        super(CNNModel, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, num_filters, kernel_size=3, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(num_filters, num_filters * 2, kernel_size=3, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(num_filters * 2, num_filters * 4, kernel_size=3, padding=0)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4 = nn.Conv2d(num_filters * 4, num_filters * 2, kernel_size=3, padding=0)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate size after convolutions: 150->148->74->72->36->34->17->15->7
        self.fc1 = nn.Linear(num_filters * 2 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 6)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # No softmax here, CrossEntropyLoss handles it
        
        return x