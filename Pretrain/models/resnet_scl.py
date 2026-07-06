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


class SupCL_ResNet34(torch.nn.Module):
    def __init__(self, feature_dim=128):
        super(SupCL_ResNet34, self).__init__()

        model = models.resnet34()
        model.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )
        self.model = model

    def forward(self, x):
        if x.shape[1] != 3:
            x = torch.cat([x, x, x], dim=1)
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
