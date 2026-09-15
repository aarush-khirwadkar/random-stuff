"""CIFAR-10 data: two-view SSL augmentations and plain eval transforms."""
import torchvision.transforms as T
from torchvision.datasets import CIFAR10

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


class TwoViewTransform:
    """Return two independently augmented views of the same image."""
    def __init__(self, base):
        self.base = base

    def __call__(self, x):
        return self.base(x), self.base(x)


def ssl_transform():
    # The SimCLR-style augmentation family: crop + color jitter are the key pieces.
    return T.Compose([
        T.RandomResizedCrop(32, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])


def eval_transform():
    return T.Compose([T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])


def get_ssl_dataset(root="./data", train=True):
    return CIFAR10(root, train=train, download=True,
                   transform=TwoViewTransform(ssl_transform()))


def get_eval_datasets(root="./data"):
    train = CIFAR10(root, train=True, download=True, transform=eval_transform())
    test = CIFAR10(root, train=False, download=True, transform=eval_transform())
    return train, test
