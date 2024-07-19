from typing import Dict
import flwr as fl
from typing import Dict, Tuple

from flwr.common import Config, NDArrays, Scalar
from torch.utils.data.dataset import Dataset
from .base_client import BaseClient
from flwr.server.strategy import FedProx
from utils import (
    build_model,
    build_trainer,
    build_optimizer,
    build_criterion,
    get_parameters,
)
from data.data_manager import Next_stream_data


class FedProxClient(BaseClient):
    def __init__(
        self, train_set: Dataset, test_set: Dataset, config: Dict, idx: int
    ) -> None:
        super().__init__(train_set, test_set, config, idx)

    def fit(
        self, parameters: NDArrays, training_ins: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """train the local models accoring to instructions

        Args:
            parameters (NDArrays): the given parameters
            training_ins (Dict[str, Scalar]): training instructions

        Returns:
            Tuple[NDArrays, int, Dict[str, Scalar]]: the trained parameters, the number of sample trained, the training statistics.
        """
        # step 1: receive the parameters from server
        self.set_parameters(parameters)

        # step 2: train the model according to training instructions
        optimizer = build_optimizer(
            self.config["optimizer"],
            self.model.parameters(),
            self.config["optimizer_kwargs"],
        )
        criterion = build_criterion(self.config["criterion"])
        training_loss = self.trainer.train(
            self.model,
            parameters,
            criterion,
            self.train_loader,
            optimizer,
            self.config["local_epoch"],
            self.config["device"],
            self.config["proximal_mu"],
        )
        # step 3: return the training results
        parameters_prime = self.get_parameters(None)
        num_examples_train = len(self.train_loader.dataset)
        train_status = {"training_loss": training_loss}

        # step 4: get the next stream of data
        self.train_loader, self.test_loader = Next_stream_data(
            self.train_set, self.config, self.idx
        )
        return parameters_prime, num_examples_train, train_status
