import torch
import numpy as np
import random
from typing import List, Iterator
import torch.nn as nn
from models.moe import CNN, CNN_moe_noise
import torch.optim as optim
from engine.base_trainer import BaseCVTrainer, BaseRegressionTrainer, BaseTrainer
from engine.fedprox_trainer import FedProxCVTrainer

# model list
MODEL_LIST = {
    "cnn": CNN,
    "cnn_moe": CNN_moe_noise
}

# trainer list (nested dictionary)
# we only consider CV or NLP task
TRAINER_LIST = {
    "CV": {
        "FedAvg": BaseCVTrainer,
        "FedFT": BaseCVTrainer,
        "FedProx": FedProxCVTrainer,
        "FedBN": BaseCVTrainer,
        "pFedMe": BaseCVTrainer,
        "FedHist": BaseCVTrainer,
        "Standalone": BaseCVTrainer,
    },
    "Regression": {
        "FedAvg": BaseRegressionTrainer,
        "FedProx": BaseRegressionTrainer,  # need modification
        "FedBN": BaseRegressionTrainer,
        "pFedMe": BaseRegressionTrainer,
        "FedHist": BaseRegressionTrainer,
        "Standalone": BaseRegressionTrainer,
    },
}


# utility functions
def seed_everything(seed: int) -> None:
    """Seed everything for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def build_model(model_name: str, model_kwargs: dict) -> nn.Module:
    """build model according to the given model_name

    Args:
        model_name (str): model name
        model_kwargs (dict): the model keyword arguments
    """
    model: nn.Module = MODEL_LIST[model_name](**model_kwargs)
    return model


def build_optimizer(
    optimizer_name: str, parameters: Iterator[nn.Parameter], optimizer_kwargs: dict
) -> optim.Optimizer:
    """build optimizer according to the given parameters

    Args:
        optimizer_name (str): optimizer name
        parameters (Iterator[nn.Parameter]): the parameters need for optimization
        optimizer_kwargs (dict): optimzier keyword arguments (such as lr and weight_decay)

    Returns:
        optim.Optimizer: the given optimizer
    """
    optimizer: optim.Optimizer = getattr(optim, optimizer_name)(
        params=parameters, **optimizer_kwargs
    )
    return optimizer


def build_trainer(task: str, algorithm: str) -> BaseTrainer:
    """build a trainer according to task and algorithm

    Args:
        task (str): -
        algorithm (str): -
    Returns:
        BaseTrainer: the trainer
    """
    return TRAINER_LIST[task][algorithm]()


def build_criterion(criterion: str) -> nn.Module:
    """build a criterion according the criterion name

    Args:
        criterion (str): criterion

    Returns:
        nn.Module: criterion
    """
    return getattr(nn, criterion)()


def get_parameters(net: nn.Module) -> List[np.ndarray]:
    """get parameters (ndarray format) of a network

    Args:
        net (nn.Module): the given network

    Returns:
        List[np.ndarray]: the returned parameters
    """
    return [val.cpu().numpy() for _, val in net.state_dict().items()]
