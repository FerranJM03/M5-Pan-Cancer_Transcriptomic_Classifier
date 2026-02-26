import torch
import torch.nn as nn
import torch.nn.functional as F

class CancerNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CancerNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)