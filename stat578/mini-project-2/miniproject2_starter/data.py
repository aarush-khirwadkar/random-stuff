"""CIFAR-10 for masked image modeling.

Masked modeling needs far less augmentation than contrastive learning did: there
are no positive pairs to construct, so a crop and a flip suffice.
"""
import torchvision.transforms as T
from torchvision.datasets import CIFAR10

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def mim_transform():
    """Augmentation for masked-image-modeling pretraining."""
    return T.Compose([
        T.RandomResizedCrop(32, scale=(0.6, 1.0)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])


def eval_transform():
    return T.Compose([T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])


def get_pretrain_dataset(root="./data", train=True):
    return CIFAR10(root, train=train, download=True, transform=mim_transform())


def get_eval_datasets(root="./data"):
    train = CIFAR10(root, train=True, download=True, transform=eval_transform())
    test = CIFAR10(root, train=False, download=True, transform=eval_transform())
    return train, test


def denormalize(x):
    """Undo normalization so reconstructions can be viewed as images."""
    import torch
    mean = torch.tensor(CIFAR_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR_STD, device=x.device).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0, 1)
