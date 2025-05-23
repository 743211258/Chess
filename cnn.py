import torch
import torch.nn.functional as F
from torch import nn

class CNN(nn.Module):
    def __init__(self, kernel_size, num_kernel_in_first_layer, num_kernel_in_second_layer, padding):
        super(CNN, self).__init__()
        self.kernel_size = kernel_size
        self.num_kernel_in_first_layer = num_kernel_in_first_layer
        self.num_kernel_in_second_layer = num_kernel_in_second_layer
        self.padding = padding
        self.cnn_one = nn.Conv2d(in_channels = num_kernel_in_first_layer,
                                 out_channels = num_kernel_in_second_layer,
                                 kernel_size = kernel_size,
                                 padding = padding)
        
        self.bn1 = nn.BatchNorm2d(num_kernel_in_second_layer)
        self.dropout1 = nn.Dropout2d(0.3)

        self.cnn_two = nn.Conv2d(in_channels = num_kernel_in_second_layer,
                                 out_channels = num_kernel_in_second_layer * 2,
                                 kernel_size = kernel_size,
                                 padding = padding)
        
        self.bn2 = nn.BatchNorm2d(num_kernel_in_second_layer * 2)
        self.dropout2 = nn.Dropout2d(0.3)

        self.fully_connect_one = nn.Linear(num_kernel_in_second_layer * 2 * 8 * 8,
                                           num_kernel_in_second_layer * 2)
        
        self.bn3 = nn.BatchNorm1d(num_kernel_in_second_layer * 2)

        self.fully_connect_two = nn.Linear(num_kernel_in_second_layer * 2, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.cnn_one(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.cnn_two(x)))
        x = self.dropout2(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.bn3(self.fully_connect_one(x)))
        return self.fully_connect_two(x)

        

