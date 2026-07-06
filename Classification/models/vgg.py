import torch
from scipy.cluster.hierarchy import weighted
from torch import nn
from torchvision import models


class VGGNet(torch.nn.Module):
    def __init__(self, num_classes=3):
        super(VGGNet, self).__init__()
        model = models.vgg16(weights=None)
        model.classifier[-1] = nn.Linear(4096, num_classes)
        self.model = model

    def forward(self, x):
        # x = torch.cat([x, x, x], 1)
        out = self.model(x)
        return out