import torch
from scipy.cluster.hierarchy import weighted
from torch import nn
from torchvision import models


class ConvNeXt_base(torch.nn.Module):
    def __init__(self, num_classes=3):
        super(ConvNeXt_base, self).__init__()

        model = models.convnext_base(pretrained=False)
        model.classifier[2] = nn.Linear(1024, num_classes)
        self.model = model

    def forward(self, x):
        # x = torch.cat([x, x, x], 1)
        out = self.model(x)
        return out

class ConvNeXt_tiny(torch.nn.Module):
    def __init__(self, num_classes=3):
        super(ConvNeXt_tiny, self).__init__()

        model = models.convnext_tiny(pretrained=False)
        model.classifier[2] = nn.Linear(768, num_classes)
        self.model = model

    def forward(self, x):
        # x = torch.cat([x, x, x], 1)
        out = self.model(x)
        return out


class ConvNeXt_small(torch.nn.Module):
    def __init__(self, num_classes=3):
        super(ConvNeXt_small, self).__init__()

        model = models.convnext_small(pretrained=False)
        model.classifier[2] = nn.Linear(768, num_classes)
        self.model = model
    
    def forward(self, x):
        out = self.model(x)
        return out


class ConvNeXt_large(torch.nn.Module):
    def __init__(self, num_classes=3):
        super(ConvNeXt_large, self).__init__()

        model = models.convnext_large(pretrained=False)
        model.classifier[2] = nn.Linear(1536, num_classes)
        self.model = model

    def forward(self, x):
        # x = torch.cat([x, x, x], 1)
        out = self.model(x)
        return out