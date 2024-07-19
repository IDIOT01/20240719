# definitios of datasets
from torchvision.datasets import CIFAR10, CIFAR100, Flowers102, FashionMNIST, MNIST
from torch.utils.data import ConcatDataset, Dataset, TensorDataset
from torchvision.transforms import (
    ToTensor,
    Lambda,
    ToPILImage,
    RandAugment,
    Normalize,
    Resize,
    Compose,
)
import torch.nn as nn
from .data_manager import dirichlet_data_partition
from typing import Dict, List

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DATASET_LIST: Dict[str, Dataset] = {
    "CIFAR10": CIFAR10,
    "CIFAR100": CIFAR100,
    "FashionMNIST": FashionMNIST,
    "MNIST": MNIST,
    # add more datasets
}


TRANSFORM_LIST: Dict[str, nn.Module] = {
    "CIFAR10": Compose(
        [
            Resize(64),
            ToTensor(),
            Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    ),
    "CIFAR100": Compose(
        [
            Resize(64),
            ToTensor(),
            Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    ),
    "FashionMNIST": Compose(
        [
            Resize(64),
            ToTensor(),
            Lambda(
                lambda x: x.repeat(3, 1, 1)
            ),  # Repeat the single channel image 3 times
            Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    ),
    "MNIST": Compose(
        [
            Resize(64),
            ToTensor(),
            Lambda(
                lambda x: x.repeat(3, 1, 1)
            ),  # Repeat the single channel image 3 times
            Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    ),
}


def load_dataset(
    dataset_name: str, data_root: str, num_clients: int, alpha: float
) -> List[Dataset]:

    if dataset_name in ["CIFAR10", "CIFAR100", "FashionMNIST", "MNIST"]:
        train_set = DATASET_LIST[dataset_name](
            root=data_root,
            train=True,
            transform=TRANSFORM_LIST[dataset_name],
            download=True,
        )
        test_set = DATASET_LIST[dataset_name](
            root=data_root,
            train=False,
            transform=TRANSFORM_LIST[dataset_name],
            download=True,
        )

        merged_dataset = train_set  # ConcatDataset([train_set, test_set])
    else:
        raise NotImplementedError(
            "The dataset {} is currently not supported".format(dataset_name)
        )

    datasets_client = dirichlet_data_partition(merged_dataset, num_clients, alpha)

    return datasets_client
