import torch
from scipy.cluster.hierarchy import weighted
from torch import nn
from torchvision import models


class ResNet_18(torch.nn.Module):
    def __init__(self, num_classes=3):
        super(ResNet_18, self).__init__()

        model = models.resnet18()
        model.fc = nn.Linear(512, num_classes)
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out

class ResNet_34(torch.nn.Module):
    def __init__(self, num_classes=3):
        super(ResNet_34, self).__init__()

        model = models.resnet34()
        model.fc = nn.Linear(512, num_classes)
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out

class ResNet_50(torch.nn.Module):
    def __init__(self, num_classes=3):
        super(ResNet_50, self).__init__()

        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(2048, num_classes)
        self.model = model

    def forward(self, x):
        # x = torch.cat([x, x, x], 1)
        out = self.model(x)
        return out

class ResNet_101(torch.nn.Module):
    def __init__(self, num_classes=3):
        super(ResNet_101, self).__init__()

        model = models.resnet101()
        model.fc = nn.Linear(2048, num_classes)
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out
