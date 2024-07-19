from typing import Callable, Union, OrderedDict
from .base_strategy import BaseStrategy, weighted_metrics_avg
import numpy as np
import torch

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from typing import Dict, List, Optional, Tuple
import flwr as fl
from utils import build_model, get_parameters


class FedBNStrategy(BaseStrategy):
    def __init__(self, config: Dict) -> None:
        super().__init__(config)

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        net = build_model(self.config["model_name"], self.config["model_kwargs"])
        ndarrays = [
            val.cpu().numpy()
            for name, val in net.state_dict().items()
            if "bn" not in name
        ]
        return fl.common.ndarrays_to_parameters(ndarrays)
