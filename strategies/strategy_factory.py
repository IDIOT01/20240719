from .base_strategy import BaseStrategy
from .fedhist_strategy import FedHISTStrategy
from .pfedme_strategy import pFedMeStrategy
from .standalone_strategy import StandaloneStrategy
from .fedbn_strategy import FedBNStrategy
from .fedft_strategy import FedFTStrategy
from typing import Type

STRATEGY_LIST = {
    "FedAvg": BaseStrategy,
    "FedBN": FedBNStrategy,
    # add other
    "FedProx": BaseStrategy,
    "pFedMe": pFedMeStrategy,
    "FedHist": FedHISTStrategy,
    "Standalone": StandaloneStrategy,
    "FedFT": FedFTStrategy,
}


def strategy_factory(algorithm: str) -> Type[BaseStrategy]:
    """get strategy class according to different algorithms

    Args:
        algorithm (str): your fl algorithm

    Returns:
        type[BaseStrategy]: the strategy class
    """
    return STRATEGY_LIST[algorithm]
