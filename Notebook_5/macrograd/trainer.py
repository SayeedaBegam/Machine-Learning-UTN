from abc import ABC, abstractmethod
from typing import Tuple
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm  # can be used to show progress of a for-loop


class Trainer(ABC):
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        data_loaders: Tuple[DataLoader, DataLoader],
        optim,
    ) -> None:
        self.model = model
        self.train_data, self.val_data = data_loaders
        self.crit = loss_fn
        self.optim = optim

        self.train_losses: list[float] = []
        self.train_accs: list[float] = []

        self.val_losses: list[float] = []
        self.val_accs: list[float] = []

    def fit(self, epochs: int):
        for epoch in range(epochs):
            self.train_epoch(epoch)
            self.val_epoch(epoch)
        print(
            f"After Training:\n"
            f"Train Loss: {self.train_losses[-1]:.2f},\n"
            f"Train Acc: {self.train_accs[-1]:.2f},\n"
            f"Val Loss: {self.val_losses[-1]:.2f},\n"
            f"Val Acc: {self.val_accs[-1]:.2f}"
        )

    @abstractmethod
    def train_epoch(self, epoch: int):
        ...

    @abstractmethod
    def val_epoch(self, epoch: int):
        ...


class BinaryTrainer(Trainer):

    ####################################################################
    # YOUR CODE
    ####################################################################
    pass
    ####################################################################
    # END OF YOUR CODE
    ####################################################################


class CategoricalTrainer(Trainer):
    # Hints: dont forget one-hot encoding
    def __init__(
        self, model, loss_fn, data_loaders, optim, num_classes
    ) -> None:
        super().__init__(model, loss_fn, data_loaders, optim)
        self.num_classes = num_classes

    ####################################################################
    # YOUR CODE
    ####################################################################

    ####################################################################
    # END OF YOUR CODE
    ####################################################################
