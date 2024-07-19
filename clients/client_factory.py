from base_client import BaseClient
from fedbn_client import FedBNClient
from fedhist_client import FedHISTClient
from fedstream_client import FedStreamClient
from fedprox_client import FedProxClient
from pfedme_client import pfedmeClient
from standalone_client import StandaloneClient
from typing import Type

CLIENT_LIST = {
    "FedAvg": BaseClient,
    "FedBN": FedBNClient,
    # add other
    "FedProx": FedProxClient,
    "pFedMe": pfedmeClient,
    "FedHist": FedHISTClient,
    "Standalone": StandaloneClient,
}


def client_factory(algorithm: str) -> Type[BaseClient]:
    """get client class according to different algorithms

    Args:
        algorithm (str): your fl algorithm

    Returns:
        type[BaseClient]: the client class
    """
    return CLIENT_LIST[algorithm]
