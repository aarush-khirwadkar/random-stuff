"""Encoder (CIFAR ResNet-18), projector, and predictor MLPs."""
import torch.nn as nn
from torchvision.models import resnet18


def cifar_encoder():
    """ResNet-18 adapted for 32x32 CIFAR: 3x3 stem, no max-pool, no classifier.
    Returns (network, feature_dim). The network maps an image to a 512-d vector."""
    net = resnet18(weights=None)
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    net.fc = nn.Identity()
    return net, 512


def mlp(sizes, last_bn=False):
    """Build an MLP: Linear-BN-ReLU blocks, with an optional (affine-free) BN on the output."""
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
        if i < len(sizes) - 2:
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        elif last_bn:
            layers.append(nn.BatchNorm1d(sizes[i + 1], affine=False))
    return nn.Sequential(*layers)


class Projector(nn.Module):
    """g: maps encoder features (kept for downstream) into the loss space z."""
    def __init__(self, in_dim=512, hidden=512, out=128, last_bn=True):
        super().__init__()
        self.net = mlp([in_dim, hidden, out], last_bn=last_bn)

    def forward(self, x):
        return self.net(x)


class Predictor(nn.Module):
    """q: bottleneck MLP used by BYOL/SimSiam (online branch only)."""
    def __init__(self, dim=128, hidden=512):
        super().__init__()
        self.net = mlp([dim, hidden, dim], last_bn=False)

    def forward(self, x):
        return self.net(x)
